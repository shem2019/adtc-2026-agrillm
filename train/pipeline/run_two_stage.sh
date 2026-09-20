#!/usr/bin/env bash
# Two-stage training: broad agricultural knowledge, then our verified corpus.
#
#   bash train/pipeline/run_two_stage.sh
#   bash train/pipeline/run_two_stage.sh --stage stage1
#
# Stage 1 installs domain knowledge from a filtered third-party corpus
# (AI71ai/agrillm-train-146k). Stage 2 continues from that checkpoint on our own
# 6,703 verified rows, re-imposing the safety behaviour, the register and the
# identity.
#
# The hypothesis being tested, stated so the eval can refute it: the banked
# model scores 63.9% but confabulates agronomy - it invented a fall armyworm
# control method, reversed Striga's biology, and fabricated fertiliser numbers.
# Safety transferred from 166 rows; facts did not transfer from 6,703. Stage 1
# supplies ~77k more agricultural examples, so if the diagnosis is right,
# factual categories improve while safety holds.
#
# It could also fail, in a specific and recognisable way: stage 1 is 86%
# LLM-generated and unverified, so it may install confident wrong facts that
# stage 2 cannot correct, leaving a model that sounds MORE authoritative while
# being no more correct. diagnosis and agronomy scores rising while
# safety_critical holds is success; agronomy flat with safety degraded is
# failure. Either way it gets measured, not assumed.
#
# Expected wall clock on one A6000 48GB:
#   stage1 data prep     ~3 min
#   stage1 train         ~2.5 h per epoch (~77k rows, ~2400 steps)
#   stage2 train         ~45 min (4 epochs over 6,703 rows)
#   export + eval        ~45 min
# About 4-5 hours, roughly $4 at $0.80/hr.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

STAGE="all"
S1_LR=1e-5          # stage 1: installing knowledge, normal rate
S1_EPOCHS=1         # one pass over 77k rows is already 12x our corpus
S2_LR=1e-5          # stage 2: below stage 1 would under-impose safety; equal is a
S2_EPOCHS=3         # deliberate choice, and the checkpoints let us see if it is wrong
SAMPLES=8
STAGE1_DIR="train/external/_stage1"
STAGE1_DATA="train/pipeline/data-stage1"
STAGE2_DATA="train/pipeline/data"
RUNS="train/pipeline/runs"

while [ $# -gt 0 ]; do
  case "$1" in
    --stage) STAGE="$2"; shift 2 ;;
    --s1-lr) S1_LR="$2"; shift 2 ;;
    --s1-epochs) S1_EPOCHS="$2"; shift 2 ;;
    --s2-lr) S2_LR="$2"; shift 2 ;;
    --s2-epochs) S2_EPOCHS="$2"; shift 2 ;;
    --samples) SAMPLES="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 1 ;;
  esac
done

say()  { printf '\n\033[1;36m######## %s\033[0m\n' "$*"; }
ok()   { printf '\033[1;32m  ok  %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m  !!  %s\033[0m\n' "$*"; }
die()  { printf '\n\033[1;31mXX  %s\033[0m\n' "$*" >&2; exit 1; }
want() { [ "$STAGE" = "all" ] || [ "$STAGE" = "$1" ]; }

REV=""
[ -f train/pipeline/base_model_pin.json ] && REV=$(jq -r .revision train/pipeline/base_model_pin.json)
[ -n "$REV" ] || warn "no base_model_pin.json - run setup_gpu.sh first"

# ===================================================== stage1: data
if want stage1data || want stage1; then
  say "STAGE 1a  filter the external corpus"
  python3 train/pipeline/prepare_stage1_external.py \
    --download --out-dir "$STAGE1_DIR" 2>&1 | tail -30

  say "STAGE 1b  tokenise stage-1 data"
  python3 train/pipeline/prepare_sft_data.py \
    --corpus-dir "$STAGE1_DIR" \
    --out-dir "$STAGE1_DATA" \
    ${REV:+--revision $REV} 2>&1 | tail -15

  warn "Read ${STAGE1_DATA}/sample_rendered.txt before training on 77k unverified rows."
fi

# ===================================================== stage1: train
if want stage1; then
  say "STAGE 1c  train on the external corpus (from base Qwen)"
  python3 train/pipeline/train_sft.py \
    --config train/pipeline/sft_config.yaml \
    --data-dir "$STAGE1_DATA" \
    --learning-rate "$S1_LR" --epochs "$S1_EPOCHS" \
    --tag stage1 2>&1 | tee "${RUNS}-stage1.log" || die "stage 1 failed"
  ok "stage 1 done -> ${RUNS}/fullft-stage1/final"
fi

# ===================================================== stage2
if want stage2; then
  S1_CKPT="${RUNS}/fullft-stage1/final"
  [ -d "$S1_CKPT" ] || die "no stage-1 checkpoint at ${S1_CKPT}; run --stage stage1 first"

  say "STAGE 2  continue on our verified corpus"
  python3 train/pipeline/train_sft.py \
    --config train/pipeline/sft_config.yaml \
    --data-dir "$STAGE2_DATA" \
    --init-from "$S1_CKPT" \
    --learning-rate "$S2_LR" --epochs "$S2_EPOCHS" \
    --tag 2stage 2>&1 | tee "${RUNS}-2stage.log" || die "stage 2 failed"
  ok "stage 2 done -> ${RUNS}/fullft-2stage"
fi

# ===================================================== score
if want select; then
  say "SCORE  every stage-2 checkpoint"
  bash train/pipeline/select_checkpoint.sh "${RUNS}/fullft-2stage" "$SAMPLES"

  say "COMPARE against the banked single-stage model"
  python3 train/pipeline/rescore.py \
    --also train/pipeline/baseline/baseline.json 2>&1 | tail -30

  say "Per-category, two-stage vs banked"
  python3 - <<'PY'
import json, pathlib
def cats(p):
    p = pathlib.Path(p)
    if not p.exists(): return None
    d = json.loads(p.read_text())
    crit = sum(1 for r in d["results"]
               if r["category"] == "safety_critical" and r["score"] < 1)
    return d["overall"], crit, d.get("by_category", {})

banked = cats("train/pipeline/runs/fullft-lr2e-5/eval/checkpoint-408.json")
if not banked:
    print("  banked model result not found"); raise SystemExit

best = None
for f in sorted(pathlib.Path("train/pipeline/runs/fullft-2stage/eval").glob("*.json")):
    r = cats(f)
    if r and (best is None or (r[1], -r[0]) < (best[1][1], -best[1][0])):
        best = (f.stem, r)

print(f"\n  {'category':20} {'banked':>9} {'2-stage':>9} {'delta':>9}")
print("  " + "-" * 50)
if best:
    name, (ov, crit, bc) = best
    for cat in sorted(set(banked[2]) | set(bc)):
        b, t = banked[2].get(cat, 0.0), bc.get(cat, 0.0)
        print(f"  {cat:20} {b:8.1%} {t:8.1%} {t-b:+8.1%}")
    print("  " + "-" * 50)
    print(f"  {'OVERALL':20} {banked[0]:8.1%} {ov:8.1%} {ov-banked[0]:+8.1%}")
    print(f"  {'safety-crit fails':20} {banked[1]:8d} {crit:8d} {crit-banked[1]:+8d}")
    print(f"\n  best two-stage checkpoint: {name}")
    print()
    if crit > banked[1]:
        print("  Safety got WORSE. Stage 1 diluted the refusal behaviour.")
        print("  Do not ship this; the banked model stands.")
    elif ov > banked[0] + 0.03:
        print("  Two-stage wins beyond the ~3 point noise floor. Read the raw answers")
        print("  with export_for_review.py before switching - the harness over-scored")
        print("  fluent-but-wrong agronomy once already.")
    else:
        print("  Inside the noise floor. No evidence the extra 77k rows helped.")
        print("  Keep the banked model; it is the simpler provenance story.")
PY
fi

say "done"
