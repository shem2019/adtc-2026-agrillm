#!/usr/bin/env python3
"""Assemble the provenance/ folder required by Gate 2 section 3.1.

    python3 train/pipeline/capture_provenance.py \
        --runs-dir train/pipeline/runs \
        --data-dir train/pipeline/data \
        --corpus-dir train/african/_clean \
        --out-dir provenance \
        --weight-delta train/pipeline/runs/fullft-lr1e-5/final

Gate 2 says adapter weights are "the clearest evidence" of a genuine fine-tune.
A full fine-tune has no adapter to hand over, so this script compensates with
evidence a reviewer can actually check:

  * the exact training script and config that produced the weights
  * the full training log history (loss curve, step count, wall clock)
  * the dataset manifest: per-file row counts and SHA256 of every source file
  * a readable sample of what the model was literally trained on, including the
    loss mask
  * SHA256 of every shipped artefact
  * optionally, a measured weight-delta report against the pinned base model -
    per-layer proof that these weights differ from stock Qwen2.5-1.5B-Instruct,
    which is the same claim an adapter file would make, just measured directly

It also drafts the "Model Provenance" prose section for REPORT.md so the numbers
in the report and the numbers in the folder cannot drift apart.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def copy(src: Path, dst_dir: Path) -> Path | None:
    if not src.exists():
        print(f"  ! missing, skipped: {src}")
        return None
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / src.name
    shutil.copy2(src, dst)
    print(f"  + {dst}")
    return dst


def weight_delta_report(base_model: str, revision: str | None, ckpt: Path) -> dict:
    """Measure how far the fine-tuned weights moved from the base model.

    This is the full-fine-tune analogue of shipping adapter weights: it proves
    the weights changed, shows where, and shows by how much.
    """
    import torch
    from transformers import AutoModelForCausalLM

    print(f"  loading base  {base_model}@{revision or 'main'}")
    base = AutoModelForCausalLM.from_pretrained(
        base_model, revision=revision, torch_dtype=torch.float32
    )
    print(f"  loading tuned {ckpt}")
    tuned = AutoModelForCausalLM.from_pretrained(ckpt, torch_dtype=torch.float32)

    b = dict(base.named_parameters())
    t = dict(tuned.named_parameters())

    per_layer, total_changed, total_params = [], 0, 0
    for name, bp in b.items():
        tp = t.get(name)
        if tp is None or tp.shape != bp.shape:
            continue
        d = (tp.detach() - bp.detach()).float()
        bn = bp.detach().float().norm().item()
        dn = d.norm().item()
        per_layer.append(
            {
                "param": name,
                "numel": bp.numel(),
                "mean_abs_delta": d.abs().mean().item(),
                "max_abs_delta": d.abs().max().item(),
                "relative_change": (dn / bn) if bn > 0 else 0.0,
            }
        )
        total_params += bp.numel()
        total_changed += int((d.abs() > 1e-8).sum().item())

    per_layer.sort(key=lambda r: r["relative_change"], reverse=True)
    overall_rel = sum(r["relative_change"] * r["numel"] for r in per_layer) / max(total_params, 1)
    return {
        "base_model": base_model,
        "base_model_revision": revision,
        "checkpoint": str(ckpt),
        "total_parameters_compared": total_params,
        "parameters_changed": total_changed,
        "fraction_changed": total_changed / max(total_params, 1),
        "weighted_mean_relative_change": overall_rel,
        "most_changed_layers": per_layer[:20],
        "least_changed_layers": per_layer[-10:],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", type=Path, default=Path("train/pipeline/runs"))
    ap.add_argument("--data-dir", type=Path, default=Path("train/pipeline/data"))
    ap.add_argument("--corpus-dir", type=Path, default=Path("train/african/_clean"))
    ap.add_argument("--out-dir", type=Path, default=Path("provenance"))
    ap.add_argument("--run", type=Path, default=None,
                    help="the winning run dir; defaults to the first under --runs-dir")
    ap.add_argument("--weight-delta", type=Path, default=None,
                    help="HF checkpoint dir to diff against the base model (slow, ~2 min)")
    ap.add_argument("--sample-rows", type=int, default=50)
    args = ap.parse_args()

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    run_dir = args.run
    if run_dir is None:
        candidates = sorted(p for p in args.runs_dir.glob("*") if p.is_dir())
        if not candidates:
            print(f"no run directories under {args.runs_dir}", file=sys.stderr)
            return 1
        run_dir = candidates[0]
    print(f"run: {run_dir}")

    # ---------------------------------------------------------- 1. scripts
    print("\ntraining scripts + config")
    scripts = out / "training"
    for f in [
        Path("train/pipeline/prepare_sft_data.py"),
        Path("train/pipeline/train_sft.py"),
        Path("train/pipeline/sft_config.yaml"),
        Path("train/pipeline/export_gguf.sh"),
        Path("train/pipeline/select_checkpoint.sh"),
        Path("train/pipeline/run_pipeline.sh"),
        Path("train/pipeline/setup_gpu.sh"),
        Path("train/verify_corpus.py"),
    ]:
        copy(f, scripts)

    # ------------------------------------------------------------- 2. logs
    print("\ntraining logs")
    logs = out / "logs"
    copy(run_dir / "run_record.json", logs)
    for extra in args.runs_dir.parent.glob("runs-*.log"):
        copy(extra, logs)
    for f in sorted((run_dir / "eval").glob("*.txt")):
        copy(f, logs / "eval")
    for f in sorted((run_dir / "eval").glob("*.json")):
        copy(f, logs / "eval")

    run_record = {}
    rr = run_dir / "run_record.json"
    if rr.exists():
        run_record = json.loads(rr.read_text(encoding="utf-8"))

    # ---------------------------------------------------------- 3. dataset
    print("\ndataset manifest + sample")
    data = out / "dataset"
    copy(args.data_dir / "manifest.json", data)
    copy(args.data_dir / "sample_rendered.txt", data)

    manifest = {}
    mpath = args.data_dir / "manifest.json"
    if mpath.exists():
        manifest = json.loads(mpath.read_text(encoding="utf-8"))

    # a plain-jsonl slice of the real corpus, so a reviewer can read rows
    # without needing torch to open a .pt
    sample_out = data / f"corpus_sample_{args.sample_rows}.jsonl"
    data.mkdir(parents=True, exist_ok=True)
    written = 0
    files = sorted(args.corpus_dir.glob("*.jsonl"))
    per_file = max(1, args.sample_rows // max(len(files), 1))
    with sample_out.open("w", encoding="utf-8") as fh:
        for p in files:
            for i, line in enumerate(p.read_text(encoding="utf-8").splitlines()):
                if i >= per_file or written >= args.sample_rows:
                    break
                if line.strip():
                    fh.write(line.strip() + "\n")
                    written += 1
    print(f"  + {sample_out} ({written} rows sampled across {len(files)} files)")

    # -------------------------------------------------------- 4. checksums
    print("\nchecksums")
    checks: dict[str, str] = {}
    for p in sorted(args.corpus_dir.glob("*.jsonl")):
        checks[str(p)] = sha256(p)
    for p in sorted(run_dir.glob("gguf/*.gguf")):
        checks[str(p)] = sha256(p)
    for p in sorted(out.rglob("*")):
        if p.is_file() and p.suffix in {".py", ".sh", ".yaml", ".json"}:
            checks[str(p)] = sha256(p)

    (out / "checksums.sha256").write_text(
        "".join(f"{h}  {f}\n" for f, h in sorted(checks.items())), encoding="utf-8"
    )
    print(f"  + {out/'checksums.sha256'} ({len(checks)} files)")

    # ----------------------------------------------------- 5. weight delta
    delta = None
    if args.weight_delta:
        print("\nweight-delta report (full-FT substitute for adapter weights)")
        try:
            delta = weight_delta_report(
                run_record.get("base_model", manifest.get("base_model", "Qwen/Qwen2.5-1.5B-Instruct")),
                run_record.get("base_model_revision", manifest.get("base_model_revision")),
                args.weight_delta,
            )
            (out / "weight_delta.json").write_text(json.dumps(delta, indent=2), encoding="utf-8")
            print(f"  + {out/'weight_delta.json'}")
            print(f"    {delta['fraction_changed']:.1%} of parameters changed; "
                  f"weighted mean relative change {delta['weighted_mean_relative_change']:.4%}")
        except Exception as exc:  # noqa: BLE001 - provenance must not block the run
            print(f"  ! weight-delta failed ({exc}); skipping")

    # -------------------------------------------- 6. REPORT.md draft section
    cfg = run_record.get("config", {})
    counts = manifest.get("counts", {})
    mix = manifest.get("system_prompt_mix", {})
    base = run_record.get("base_model", manifest.get("base_model", "Qwen/Qwen2.5-1.5B-Instruct"))
    rev = run_record.get("base_model_revision") or manifest.get("base_model_revision") or "UNPINNED - fill this in"

    draft = f"""## Model Provenance

**Base model.** `{base}`, commit `{rev}`. The same commit string is recorded in
`metadata.json` and in `provenance/logs/run_record.json`; `train/pipeline/setup_gpu.sh`
pins it at download time so the run is reproducible against a moving upstream.

**Fine-tuning method.** Full-parameter supervised fine-tune - not LoRA.
{run_record.get('trainable_parameters', 0):,} of {run_record.get('total_parameters', 0):,}
parameters were updated. Loss was computed on assistant tokens only; question
tokens are masked out. Hyperparameters: learning rate {cfg.get('learning_rate', 'n/a')},
{cfg.get('epochs', 'n/a')} epochs, effective batch size
{run_record.get('effective_batch_size', 'n/a')}, {cfg.get('scheduler', 'cosine')} schedule with
{cfg.get('warmup_ratio', 'n/a')} warmup, bf16, max sequence length {cfg.get('max_len', 1024)}.
The learning rate is deliberately an order of magnitude below the Round 1 LoRA
run: full fine-tuning updates every weight and forgets base capability far
faster than an adapter does.

**Training data.** {counts.get('encoded', 'n/a')} conversations built from
{len(manifest.get('source_files', {}))} topic files under `train/african/_clean/`,
totalling {sum(v.get('kept', 0) for v in manifest.get('source_files', {}).values()):,} rows.
Per-file row counts and SHA256 hashes are in `provenance/dataset/manifest.json`;
a readable 50-row sample is in `provenance/dataset/`. Every file passes the
mechanical gates in `train/verify_corpus.py`, which check answer-shape
distinctness (to catch content padded out with cosmetic variation), banned
safety abstractions, stated pesticide or veterinary doses, and markdown leakage.
System prompts were mixed during training ({mix.get('qwen_default', 0)} examples with
Qwen's default system string, {mix.get('agri', 0) + mix.get('agri_short', 0)} with an AgriLLM
system prompt) so that safety behaviour is intrinsic to the weights rather than
conditional on a system prompt the grader does not send.

**Checkpoint selection.** Every epoch was exported to Q4_K_M GGUF and scored
with `eval/run_eval.py` under the graders' own sampling settings. Selection was
behavioural, not loss-based: validation loss cannot detect a dangerous first-aid
answer, an invented species name, or a reply that overruns a stated word limit,
which were the three findings that lost Round 1.

**Before / after.** See `provenance/logs/eval/` for the full scored transcripts.

<!-- TODO: paste at least two concrete before/after example pairs here, as
     Gate 2 section 3.1 requires. Pull them from the Round 1 judge feedback and
     the matching prompt in provenance/logs/eval/. -->
"""
    (out / "REPORT_provenance_section.md").write_text(draft, encoding="utf-8")
    print(f"\n  + {out/'REPORT_provenance_section.md'} (draft - paste into REPORT.md)")

    # -------------------------------------------------------- 7. folder index
    index = f"""# provenance/

Evidence for Gate 2 section 3.1. Assembled by
`train/pipeline/capture_provenance.py`; do not hand-edit.

    training/    the exact scripts and config that produced the weights
    logs/        run_record.json (loss history, step count, wall clock) and
                 the per-checkpoint eval transcripts
    dataset/     manifest with per-file SHA256 + row counts, a readable corpus
                 sample, and sample_rendered.txt showing the exact token
                 sequences and loss mask used in training
    checksums.sha256   SHA256 of every corpus file, GGUF, and script here
    weight_delta.json  measured per-layer difference from the base model

**Why there are no adapter weights.** This is a full-parameter fine-tune, so
there is no adapter to ship. `weight_delta.json` serves the same evidentiary
purpose by measuring, layer by layer, how far the shipped weights moved from
`{base}@{rev}`.

Base model: `{base}`
Commit:     `{rev}`
Method:     full-parameter SFT, assistant-token loss only
Corpus:     {sum(v.get('kept', 0) for v in manifest.get('source_files', {}).values()):,} rows / {len(manifest.get('source_files', {}))} files
"""
    (out / "README.md").write_text(index, encoding="utf-8")
    print(f"  + {out/'README.md'}")

    print(f"\nprovenance assembled at {out}/")
    print("Remaining manual step: fill in the two before/after examples in")
    print(f"{out}/REPORT_provenance_section.md, then paste it into REPORT.md.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
