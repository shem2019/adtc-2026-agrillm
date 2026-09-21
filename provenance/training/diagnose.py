#!/usr/bin/env python3
"""Find out exactly why training goes NaN. One run, no guessing.

    python3 train/pipeline/diagnose.py

Three attempts at this have each changed one thing and hoped. This instead
establishes ground truth:

  1. what code is actually checked out, and what versions are installed
  2. what dtype the model ACTUALLY loads at, under each kwarg spelling
  3. a real forward+backward loop over real batches, under four different
     numerical configurations, reporting the first micro-batch where anything
     goes non-finite - and which corpus rows were in that batch

If it is the dtype, config D survives and A does not. If it is attention
numerics, B survives and A does not. If it is one poisoned row, every config
dies on the same micro-batch index and the row is named.

Takes about three minutes. Cheaper than another failed training run.
"""
from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

HERE = Path(__file__).parent
REPO = HERE.parent.parent
IGNORE_INDEX = -100
N_MICRO = 220          # failure appeared by optimiser step 20 = 160 micro-batches
MICRO_BATCH = 4


def rule(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def sh(cmd: str) -> str:
    try:
        return subprocess.run(cmd, shell=True, cwd=REPO, capture_output=True,
                              text=True, timeout=30).stdout.strip()
    except Exception as exc:  # noqa: BLE001
        return f"<failed: {exc}>"


# ----------------------------------------------------------------- 1. environment
def check_environment() -> dict:
    rule("1. ENVIRONMENT AND CHECKED-OUT CODE")

    head = sh("git rev-parse --short HEAD")
    subject = sh("git log -1 --pretty=%s")
    dirty = sh("git status --porcelain")
    print(f"  git HEAD        : {head}  {subject}")
    print(f"  working tree    : {'DIRTY' if dirty else 'clean'}")
    if dirty:
        for line in dirty.splitlines():
            print(f"                    {line}")

    import transformers
    print(f"  torch           : {torch.__version__}  cuda={torch.version.cuda}")
    print(f"  transformers    : {transformers.__version__}")
    print(f"  cuda available  : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  device          : {torch.cuda.get_device_name(0)}")

    train_sft = (HERE / "train_sft.py").read_text(encoding="utf-8")
    markers = {
        "fp32 force+assert present": "forcing fp32" in train_sft,
        "AbortOnNaN present": "AbortOnNaN" in train_sft,
        "dtype= attempted first": "dtype=torch.float32, **_load_kwargs" in train_sft,
    }
    print("\n  code markers (are the fixes actually here?):")
    for name, present in markers.items():
        print(f"    [{'x' if present else ' '}] {name}")
    if not all(markers.values()):
        print("\n  !! The checked-out train_sft.py is NOT the fixed version.")
        print("     git checkout -- train/pipeline/sft_config.yaml && git pull")
    return markers


# ----------------------------------------------------------------- 2. dtype truth
def check_dtype(base_model: str, revision: str | None) -> None:
    rule("2. WHAT DTYPE DOES THE MODEL ACTUALLY LOAD AT?")
    from transformers import AutoModelForCausalLM

    cfg_dtype = None
    try:
        from transformers import AutoConfig
        c = AutoConfig.from_pretrained(base_model, revision=revision)
        cfg_dtype = getattr(c, "torch_dtype", None) or getattr(c, "dtype", None)
    except Exception:  # noqa: BLE001
        pass
    print(f"  dtype declared in the model's own config.json : {cfg_dtype}")
    print("  (if a kwarg is ignored, THIS is what you silently get)\n")

    for label, kwargs in [
        ("dtype=torch.float32", {"dtype": torch.float32}),
        ("torch_dtype=torch.float32", {"torch_dtype": torch.float32}),
        ("no dtype kwarg at all", {}),
    ]:
        try:
            m = AutoModelForCausalLM.from_pretrained(base_model, revision=revision, **kwargs)
            got = next(m.parameters()).dtype
            verdict = "OK" if got == torch.float32 else "IGNORED -> got config default"
            print(f"  {label:28} -> {str(got):16} {verdict}")
            del m
        except TypeError as exc:
            print(f"  {label:28} -> TypeError: {exc}")
        except Exception as exc:  # noqa: BLE001
            print(f"  {label:28} -> {type(exc).__name__}: {exc}")
        torch.cuda.empty_cache()


# ----------------------------------------------------------------- 3. the real loop
class Collator:
    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, feats):
        n = max(len(f["input_ids"]) for f in feats)
        ids, lab, att, src = [], [], [], []
        for f in feats:
            i = list(f["input_ids"])
            l = list(f["labels"])
            p = n - len(i)
            ids.append(i + [self.pad_id] * p)
            lab.append(l + [IGNORE_INDEX] * p)
            att.append([1] * len(i) + [0] * p)
            src.append(f.get("source", "?"))
        return (
            {
                "input_ids": torch.tensor(ids, dtype=torch.long),
                "labels": torch.tensor(lab, dtype=torch.long),
                "attention_mask": torch.tensor(att, dtype=torch.long),
            },
            src,
        )


def run_config(label: str, base_model: str, revision: str | None, rows: list,
               pad_id: int, *, param_dtype, autocast: bool, attn: str,
               fused: bool) -> dict:
    from transformers import AutoModelForCausalLM

    print(f"\n  --- {label} ---")
    print(f"      params={param_dtype}  autocast_bf16={autocast}  attn={attn}  fused_adamw={fused}")

    try:
        model = AutoModelForCausalLM.from_pretrained(
            base_model, revision=revision, attn_implementation=attn
        )
    except Exception as exc:  # noqa: BLE001
        print(f"      load failed: {exc}")
        return {"label": label, "status": "load_failed"}

    model = model.to(param_dtype).cuda()
    model.config.use_cache = False
    model.train()
    actual = next(model.parameters()).dtype
    print(f"      actual param dtype after .to(): {actual}")

    opt = torch.optim.AdamW(model.parameters(), lr=1e-5, fused=fused)
    loader = DataLoader(rows, batch_size=MICRO_BATCH, shuffle=False,
                        collate_fn=Collator(pad_id))

    first_bad = None
    bad_sources = None
    losses = []
    for i, (batch, srcs) in enumerate(loader):
        if i >= N_MICRO:
            break
        batch = {k: v.cuda() for k, v in batch.items()}
        if autocast:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(**batch)
                loss = out.loss
        else:
            out = model(**batch)
            loss = out.loss

        lv = loss.detach().float().item()
        losses.append(lv)
        if not math.isfinite(lv):
            first_bad, bad_sources = i, srcs
            print(f"      !! LOSS non-finite at micro-batch {i}: {lv}")
            break

        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not math.isfinite(gn.item()):
            first_bad, bad_sources = i, srcs
            print(f"      !! GRAD non-finite at micro-batch {i} (loss was {lv:.4f})")
            break
        opt.step()
        opt.zero_grad(set_to_none=True)

    if first_bad is None:
        print(f"      SURVIVED all {min(N_MICRO, len(loader))} micro-batches")
        print(f"      loss: first={losses[0]:.4f}  last={losses[-1]:.4f}  "
              f"min={min(losses):.4f}  max={max(losses):.4f}")
        status = "ok"
    else:
        print(f"      rows in the failing batch: {bad_sources}")
        status = f"failed@{first_bad}"

    del model, opt
    torch.cuda.empty_cache()
    return {"label": label, "status": status, "first_bad": first_bad,
            "bad_sources": bad_sources,
            "loss_first": losses[0] if losses else None,
            "loss_last": losses[-1] if losses else None}


def main() -> int:
    markers = check_environment()

    data_dir = HERE / "data"
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    base_model = manifest.get("base_model", "Qwen/Qwen2.5-1.5B-Instruct")
    revision = manifest.get("base_model_revision")
    print(f"\n  base model      : {base_model} @ {revision}")

    check_dtype(base_model, revision)

    rule("3. REAL FORWARD+BACKWARD OVER REAL BATCHES")
    rows = torch.load(data_dir / "train.pt", weights_only=False)
    print(f"  loaded {len(rows)} training rows; running {N_MICRO} micro-batches "
          f"of {MICRO_BATCH} per configuration")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(base_model, revision=revision)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    print(f"  pad token id    : {pad_id}")

    # sanity on the data itself before blaming numerics
    vocab = len(tok)
    mx = max(max(r["input_ids"]) for r in rows[:2000])
    mn = min(min(r["input_ids"]) for r in rows[:2000])
    empty = sum(1 for r in rows[:2000] if all(l == IGNORE_INDEX for l in r["labels"]))
    print(f"  token id range  : {mn} .. {mx}   (tokenizer len {vocab})")
    print(f"  rows with no trainable tokens (first 2000): {empty}")
    if mx >= vocab:
        print("  !! token id exceeds vocab size - this alone would produce garbage")

    configs = [
        ("A  intended: fp32 params + bf16 autocast + sdpa + fused",
         dict(param_dtype=torch.float32, autocast=True, attn="sdpa", fused=True)),
        ("B  eager attention instead of sdpa",
         dict(param_dtype=torch.float32, autocast=True, attn="eager", fused=True)),
        ("C  pure fp32, no autocast at all",
         dict(param_dtype=torch.float32, autocast=False, attn="sdpa", fused=False)),
        ("D  control: bf16 params (what we think has been running)",
         dict(param_dtype=torch.bfloat16, autocast=False, attn="sdpa", fused=False)),
    ]

    results = []
    for label, kw in configs:
        results.append(run_config(label, base_model, revision, rows, pad_id, **kw))

    rule("VERDICT")
    for r in results:
        print(f"  {r['label']:58} {r['status']}")

    ok = [r for r in results if r["status"] == "ok"]
    bad = [r for r in results if str(r["status"]).startswith("failed")]

    print()
    if not ok:
        idxs = {r["first_bad"] for r in bad}
        if len(idxs) == 1:
            print(f"  Every configuration died on the SAME micro-batch ({idxs.pop()}).")
            print("  That points at the DATA, not the numerics. The rows in that batch")
            print("  were printed above - inspect them.")
        else:
            print("  Everything failed, at different points. Numerical instability")
            print("  independent of these switches. Next step is lowering the LR hard")
            print("  (1e-6) and raising warmup, or gradient checkpointing.")
    elif len(ok) == len(results):
        print("  Everything survived here. The failure is in something this script does")
        print("  not reproduce - most likely the Trainer configuration itself rather")
        print("  than the model or data.")
    else:
        print("  Working configurations:")
        for r in ok:
            print(f"    {r['label']}")
        print("  Failing configurations:")
        for r in bad:
            print(f"    {r['label']}  (first bad micro-batch {r['first_bad']})")
        print("\n  Set sft_config.yaml to match a working configuration above.")

    out = HERE / "diagnose_result.json"
    out.write_text(json.dumps({"markers": markers, "results": results}, indent=2,
                              default=str), encoding="utf-8")
    print(f"\n  written: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
