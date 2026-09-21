#!/usr/bin/env python3
"""Full-parameter supervised fine-tune of Qwen2.5-1.5B-Instruct on the AgriLLM corpus.

    python3 train/pipeline/train_sft.py --config train/pipeline/sft_config.yaml

This is a FULL fine-tune, not LoRA. Memory maths for 1.54B params on one H100 80GB:

    bf16 weights                 ~3.1 GB
    bf16 gradients               ~3.1 GB
    AdamW fp32 exp_avg + sq      ~12.3 GB
    fp32 master weights          ~6.2 GB
    ---------------------------------------
    static                       ~24.7 GB
    activations @ 1024 tok, bs16 ~4-8 GB
    ---------------------------------------
    working set                  ~30-33 GB   (of 80 GB)

So there is a lot of headroom. Batch size is set for throughput, not survival.

Deliberate choices, each with a reason:

* LEARNING RATE IS LOW (1e-5 default). Full fine-tuning updates every weight,
  so it forgets base-model capability far faster than LoRA does. The Round 1
  run used 5e-5, which is normal for LoRA and roughly 5x too hot for full FT.
  Overcooking here shows up as fluent nonsense and lost general knowledge.

* CHECKPOINT EVERY EPOCH, SELECT BEHAVIOURALLY. Validation loss does not know
  that "induce vomiting" is a disqualifying answer. Every epoch is saved so
  eval/run_eval.py can pick the winner on judge-derived criteria instead.

* save_only_model. Optimizer state for full FT is ~18 GB per checkpoint. We
  never resume, so we do not pay for it three times over.
"""
from __future__ import annotations

import argparse
import inspect
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)

IGNORE_INDEX = -100


class AbortOnNaN(TrainerCallback):
    """Stop the run the moment the loss or gradient norm goes non-finite.

    Without this, a diverged run keeps going to the last step, writes four
    checkpoints of NaN weights, and bills you for all of it. The first attempt
    at this run spent 68% of its wall clock producing loss=0.0 with
    grad_norm=nan before anyone looked at the screen.
    """

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return control
        for key in ("loss", "grad_norm", "eval_loss"):
            val = logs.get(key)
            if val is None:
                continue
            try:
                bad = not math.isfinite(float(val))
            except (TypeError, ValueError):
                continue
            if bad:
                print(f"\n!! {key}={val} at step {state.global_step}. Aborting.", file=sys.stderr)
                print("   Non-finite loss means the weights are already ruined; every", file=sys.stderr)
                print("   further step is wasted GPU time.", file=sys.stderr)
                print("   Measured cause on this stack (see diagnose.py): SDPA attention", file=sys.stderr)
                print("   combined with bf16 compute produces non-finite grads on", file=sys.stderr)
                print("   right-padded batches. Check attn_implementation is 'eager'.", file=sys.stderr)
                print("   If it already is, run: python3 train/pipeline/diagnose.py", file=sys.stderr)
                control.should_training_stop = True
        return control


class PackedJsonlDataset(Dataset):
    """Wraps the list-of-dicts produced by prepare_sft_data.py."""

    def __init__(self, path: Path):
        self.rows = torch.load(path, weights_only=False)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        r = self.rows[idx]
        return {"input_ids": r["input_ids"], "labels": r["labels"]}


class PadCollator:
    """Right-pad a batch. Labels pad with -100 so padding never contributes loss."""

    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, features: list[dict]) -> dict:
        max_len = max(len(f["input_ids"]) for f in features)
        input_ids, labels, attn = [], [], []
        for f in features:
            ids = list(f["input_ids"])
            lab = list(f["labels"])
            pad = max_len - len(ids)
            input_ids.append(ids + [self.pad_token_id] * pad)
            labels.append(lab + [IGNORE_INDEX] * pad)
            attn.append([1] * len(ids) + [0] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
        }


def build_training_args(cfg: dict, out_dir: Path) -> TrainingArguments:
    """Construct TrainingArguments, tolerating transformers API drift.

    `evaluation_strategy` was renamed to `eval_strategy` in 4.46, and
    `save_only_model` did not exist before 4.38. Rather than pin a transformers
    version and hope the rented box agrees, we introspect the signature and drop
    anything unsupported, printing what we dropped.
    """
    wanted = {
        "output_dir": str(out_dir),
        "overwrite_output_dir": True,
        "num_train_epochs": cfg["epochs"],
        "per_device_train_batch_size": cfg["batch_size"],
        "per_device_eval_batch_size": cfg["batch_size"],
        "gradient_accumulation_steps": cfg["grad_accum"],
        "learning_rate": cfg["learning_rate"],
        "lr_scheduler_type": cfg.get("scheduler", "cosine"),
        # warmup_ratio was dropped as unsupported by a newer transformers build,
        # which silently removed warmup entirely. warmup_steps is the older,
        # still-accepted spelling; build_training_args keeps whichever survives.
        "warmup_ratio": cfg.get("warmup_ratio", 0.05),
        "warmup_steps": cfg.get("warmup_steps", 0) or None,
        "weight_decay": cfg.get("weight_decay", 0.01),
        "max_grad_norm": cfg.get("max_grad_norm", 1.0),
        "bf16": True,
        "fp16": False,
        "logging_steps": cfg.get("logging_steps", 10),
        "save_strategy": "epoch",
        "save_only_model": True,
        "eval_strategy": "epoch",
        "evaluation_strategy": "epoch",  # pre-4.46 name; one of the two survives
        "save_total_limit": cfg.get("save_total_limit", 10),
        "seed": cfg.get("seed", 1337),
        "data_seed": cfg.get("seed", 1337),
        "report_to": [],
        "gradient_checkpointing": cfg.get("gradient_checkpointing", False),
        "optim": cfg.get("optim", "adamw_torch_fused"),
        "dataloader_num_workers": cfg.get("dataloader_num_workers", 4),
        "remove_unused_columns": False,
        "group_by_length": False,
    }

    # If this build has no warmup_ratio, convert it to an explicit step count so
    # the run does not silently lose its warmup altogether.
    supported = set(inspect.signature(TrainingArguments.__init__).parameters)
    if "warmup_ratio" not in supported and "warmup_steps" in supported:
        total = cfg.get("_total_steps")
        if total:
            wanted["warmup_steps"] = max(10, int(total * cfg.get("warmup_ratio", 0.05)))
    wanted = {k: v for k, v in wanted.items() if v is not None}
    # keep only one of the eval-strategy spellings
    if "eval_strategy" in supported:
        wanted.pop("evaluation_strategy", None)
    else:
        wanted.pop("eval_strategy", None)

    dropped = [k for k in wanted if k not in supported]
    for k in dropped:
        wanted.pop(k)
    if dropped:
        print(f"[compat] transformers build does not support: {', '.join(dropped)} - dropped")

    return TrainingArguments(**wanted)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("train/pipeline/sft_config.yaml"))
    ap.add_argument("--data-dir", type=Path, default=None, help="override cfg data_dir")
    ap.add_argument("--out-dir", type=Path, default=None, help="override cfg out_dir")
    ap.add_argument("--learning-rate", type=float, default=None, help="override, for sweeps")
    ap.add_argument("--epochs", type=float, default=None, help="override, for sweeps")
    ap.add_argument("--tag", default=None, help="suffix for the run directory")
    ap.add_argument("--init-from", type=Path, default=None,
                    help="start from a local checkpoint instead of the HF base model. "
                         "This is what makes stage 2 of a two-stage run continue from "
                         "stage 1 rather than restarting from stock Qwen.")
    ap.add_argument("--batch-size", type=int, default=None,
                    help="override micro-batch size (memory), not effective batch")
    ap.add_argument("--grad-accum", type=int, default=None,
                    help="override accumulation steps; keep batch_size x grad_accum constant")
    ap.add_argument("--gradient-checkpointing", action="store_true",
                    help="recompute activations instead of storing them: large memory "
                         "saving, roughly 30 percent slower. Needed for long sequences "
                         "on a 48GB card.")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.learning_rate is not None:
        cfg["learning_rate"] = args.learning_rate
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["batch_size"] = args.batch_size
    if args.grad_accum is not None:
        cfg["grad_accum"] = args.grad_accum
    if args.gradient_checkpointing:
        cfg["gradient_checkpointing"] = True

    data_dir = args.data_dir or Path(cfg["data_dir"])
    out_dir = args.out_dir or Path(cfg["out_dir"])
    if args.tag:
        out_dir = out_dir.parent / f"{out_dir.name}-{args.tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # The allocator had 2.36 GiB reserved-but-unallocated when it OOMed. With a
    # 152k vocab the logits tensor is large and its size varies per batch with
    # sequence length, which fragments the pool badly. Expandable segments lets
    # the allocator grow a block rather than hunt for a contiguous one.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    set_seed(cfg.get("seed", 1337))

    if not torch.cuda.is_available():
        # Be specific rather than just "no CUDA". The commonest cause here is not
        # a missing GPU at all: llama.cpp's convert requirements pin torch against
        # PyTorch's CPU wheel index, so installing them replaces a working CUDA
        # build with a +cpu one and this is where it surfaces.
        print("CUDA not available to torch.", file=sys.stderr)
        print(f"  torch {torch.__version__}, compiled for CUDA {torch.version.cuda}", file=sys.stderr)
        if torch.version.cuda is None or "+cpu" in torch.__version__:
            print("  This is a CPU-only torch build. Something replaced the CUDA one.",
                  file=sys.stderr)
            print("  Fix:  pip uninstall -y torch && pip install torch", file=sys.stderr)
        else:
            print("  torch has CUDA support but cannot reach a device. Check nvidia-smi",
                  file=sys.stderr)
            print("  and whether a driver update needs a reboot to take effect.", file=sys.stderr)
        return 1
    props = torch.cuda.get_device_properties(0)
    vram_gb = props.total_memory / 1024**3
    print(f"GPU: {props.name}  {vram_gb:.1f} GB VRAM  (CUDA {torch.version.cuda}, torch {torch.__version__})")
    eff = cfg["batch_size"] * cfg["grad_accum"]
    if vram_gb < 38:
        print(f"! {vram_gb:.1f} GB is tight for full FT. Use --batch-size 2 --grad-accum 16 "
              f"--gradient-checkpointing (effective batch stays {eff}).")
    elif vram_gb < 50 and cfg["batch_size"] > 2:
        print(f"! {vram_gb:.1f} GB with batch_size {cfg['batch_size']}: long sequences may OOM "
              f"in the vocab projection. --batch-size 2 --grad-accum {eff // 2} is safer.")

    manifest_path = data_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}

    # Base-model identity resolves from three places, in order:
    #   1. sft_config.yaml, if someone set it explicitly
    #   2. base_model_pin.json, written by setup_gpu.sh at download time
    #   3. the data manifest, written by prepare_sft_data.py
    #
    # The pin file is why sft_config.yaml can stay `base_model_revision: null`
    # in git forever. Hard-coding the SHA there meant editing a tracked file on
    # every box, which then conflicted with every subsequent git pull - a
    # manual step that has to be remembered is a reproducibility bug.
    pin_path = Path("train/pipeline/base_model_pin.json")
    pin = json.loads(pin_path.read_text(encoding="utf-8")) if pin_path.exists() else {}

    base_model = (cfg.get("base_model") or pin.get("base_model")
                  or manifest.get("base_model") or "Qwen/Qwen2.5-1.5B-Instruct")
    revision = (cfg.get("base_model_revision") or pin.get("revision")
                or manifest.get("base_model_revision"))
    if revision and not cfg.get("base_model_revision"):
        src = "base_model_pin.json" if pin.get("revision") else "data manifest"
        print(f"base-model commit resolved from {src}")

    # --init-from turns this into a continuation run. The tokenizer still comes
    # from the pinned base (a checkpoint dir saved with save_only_model has no
    # tokenizer), and the provenance still records the original base model and
    # commit, plus the checkpoint we continued from.
    init_from = str(args.init_from) if args.init_from else None
    if init_from:
        if not Path(init_from).is_dir():
            print(f"--init-from path not found: {init_from}", file=sys.stderr)
            return 1
        print(f"CONTINUATION RUN")
        print(f"  initialising weights from: {init_from}")
        print(f"  original base (for tokenizer + provenance): {base_model} @ {revision}")
    else:
        print(f"base model: {base_model}" + (f" @ {revision}" if revision else " @ (unpinned)"))
    if not revision:
        print("! No base-model commit pinned. Gate 2 section 3.1 wants the exact SHA in metadata.json.")

    tok = AutoTokenizer.from_pretrained(base_model, revision=revision)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    # Load in fp32 and let bf16=True drive autocast. This is MIXED precision:
    # fp32 master weights, bf16 forward/backward.
    #
    # Loading in bf16 while also setting bf16=True is a different and much worse
    # thing - PURE bf16 training, with no fp32 master copy. AdamW then applies
    # updates directly to bf16 weights, and bf16 carries roughly three decimal
    # digits of mantissa, so an update of order 1e-6 against a weight of order
    # 1e-2 is partly or wholly lost to rounding. That is what produced
    # grad_norm=nan by step 20 and loss=0.0 thereafter on the first attempt at
    # this run, at the LOWEST learning rate in the sweep.
    # Newer transformers renamed `torch_dtype` to `dtype` and only warns about
    # the old spelling, so passing torch_dtype can be silently ignored - in which
    # case the model loads at the dtype in Qwen's own config.json, which is
    # bfloat16, and we are back to pure bf16 training without being told.
    # Try the new name, fall back to the old, then force and ASSERT the result
    # rather than trusting either kwarg.
    _load_src = init_from or base_model
    _load_kwargs = {"attn_implementation": cfg.get("attn_implementation", "sdpa")}
    if not init_from:
        _load_kwargs["revision"] = revision
    try:
        model = AutoModelForCausalLM.from_pretrained(
            _load_src, dtype=torch.float32, **_load_kwargs
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            _load_src, torch_dtype=torch.float32, **_load_kwargs
        )

    param_dtype = next(model.parameters()).dtype
    if param_dtype != torch.float32:
        print(f"! model loaded as {param_dtype} despite the request; forcing fp32")
        model = model.float()
        param_dtype = next(model.parameters()).dtype
    if param_dtype != torch.float32:
        print(f"Could not load the model in fp32 (got {param_dtype}).", file=sys.stderr)
        print("Pure bf16 full fine-tuning diverges; refusing to burn GPU time on it.",
              file=sys.stderr)
        return 1
    print(f"parameter dtype: {param_dtype} (fp32 master weights, bf16 autocast)")
    model.config.use_cache = False
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"parameters: {n_params/1e9:.3f}B total, {n_trainable/1e9:.3f}B trainable "
          f"({100*n_trainable/n_params:.1f}% - full fine-tune)")

    train_ds = PackedJsonlDataset(data_dir / "train.pt")
    val_ds = PackedJsonlDataset(data_dir / "val.pt")
    print(f"data: {len(train_ds)} train / {len(val_ds)} val examples")

    eff_batch = cfg["batch_size"] * cfg["grad_accum"]
    steps_per_epoch = math.ceil(len(train_ds) / eff_batch)
    cfg["_total_steps"] = int(steps_per_epoch * cfg["epochs"])
    targs = build_training_args(cfg, out_dir)
    print(f"effective batch {eff_batch}  ->  ~{steps_per_epoch} steps/epoch, "
          f"~{int(steps_per_epoch * cfg['epochs'])} total")
    print(f"learning rate {cfg['learning_rate']}  scheduler {cfg.get('scheduler','cosine')}  "
          f"epochs {cfg['epochs']}")

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=PadCollator(tok.pad_token_id),
        callbacks=[AbortOnNaN()],
    )

    t0 = time.time()
    result = trainer.train()
    minutes = (time.time() - t0) / 60
    print(f"\ntraining finished in {minutes:.1f} min")

    # final model in its own directory, alongside the per-epoch checkpoints
    final_dir = out_dir / "final"
    trainer.save_model(str(final_dir))
    tok.save_pretrained(str(final_dir))

    run_record = {
        "base_model": base_model,
        "base_model_revision": revision,
        "initialised_from": init_from,
        "method": "full_finetune" + ("_continuation" if init_from else ""),
        "trainable_parameters": n_trainable,
        "total_parameters": n_params,
        "config": cfg,
        "effective_batch_size": eff_batch,
        "steps_per_epoch": steps_per_epoch,
        "wall_clock_minutes": round(minutes, 2),
        "train_metrics": result.metrics,
        "log_history": trainer.state.log_history,
        "gpu": props.name,
        "torch": torch.__version__,
        "data_manifest": manifest,
    }
    (out_dir / "run_record.json").write_text(json.dumps(run_record, indent=2), encoding="utf-8")

    checkpoints = sorted(p.name for p in out_dir.glob("checkpoint-*"))
    print(f"\ncheckpoints saved: {', '.join(checkpoints) if checkpoints else '(none)'}  + final/")
    print(f"run record: {out_dir/'run_record.json'}")
    print("\nNext: export each checkpoint to GGUF and score it with eval/run_eval.py.")
    print("Pick the winner on eval score, NOT on validation loss.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
