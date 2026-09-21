#!/usr/bin/env bash
# ============================================================================
# AgriLLM — reproduce the shipped model from a bare GPU box, in one command.
#
#   git clone https://github.com/shem2019/adtc-2026-agrillm.git
#   cd adtc-2026-agrillm
#   bash train/pipeline/reproduce.sh --all
#
# That is the whole procedure. No file in the repo needs editing on the box,
# no environment variable needs exporting, and no step needs a human between
# it and the next one. Everything below was learned the expensive way during
# the Gate 2 work; each guard exists because its absence cost GPU time on a
# rented machine.
#
# Stages (run individually with --stage <name>):
#   setup     install CUDA toolkit + torch + llama.cpp, pin the base model
#   check     preflight: 12 assertions, ~30 s, must be green before training
#   stage1    train on the filtered third-party corpus   (~2.7 h on an A6000)
#   stage2    continue on our verified corpus            (~1.0 h)
#   select    export every checkpoint to GGUF and score it (~1.0 h)
#   review    write per-checkpoint raw-answer files for human reading
#
# Hardware: one NVIDIA GPU with >= 40 GB VRAM. Developed on an RTX A6000 48 GB;
# the earlier single-stage sweep ran on an H100 PCIe 80 GB. Total wall clock
# about five hours, roughly $4 at A6000 spot pricing.
# ============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

STAGE="all"
S1_LR=1e-5; S1_EPOCHS=1        # one pass over 74,697 rows is already 11x our corpus
S2_LR=2e-5; S2_EPOCHS=6        # the shipped model: checkpoint-1224 is epoch 6
SAMPLES=8                      # 8 samples/prompt; 3 is too noisy to rank checkpoints

while [ $# -gt 0 ]; do
  case "$1" in
    --all) STAGE="all"; shift ;;
    --stage) STAGE="$2"; shift 2 ;;
    --s1-lr) S1_LR="$2"; shift 2 ;;
    --s1-epochs) S1_EPOCHS="$2"; shift 2 ;;
    --s2-lr) S2_LR="$2"; shift 2 ;;
    --s2-epochs) S2_EPOCHS="$2"; shift 2 ;;
    --samples) SAMPLES="$2"; shift 2 ;;
    -h|--help) sed -n '2,30p' "$0"; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 1 ;;
  esac
done

say()  { printf '\n\033[1;36m######## %s\033[0m\n' "$*"; }
ok()   { printf '\033[1;32m  ok  %s\033[0m\n' "$*"; }
die()  { printf '\n\033[1;31mXX  %s\033[0m\n' "$*" >&2; exit 1; }
want() { [ "$STAGE" = "all" ] || [ "$STAGE" = "$1" ]; }

LOGS="$REPO_ROOT/train/pipeline/logs"; mkdir -p "$LOGS"
RUNS="train/pipeline/runs"

# ---------------------------------------------------------------- setup ----
if want setup; then
  say "SETUP  toolchain, torch, llama.cpp, base model"
  bash train/pipeline/setup_gpu.sh 2>&1 | tee "$LOGS/00-setup.log" || die "setup failed"
  ok "setup complete"
fi

# The venv must be active for everything below. Sourcing here rather than
# asking the operator to remember it is the difference between one command
# and two, and a forgotten `source` produced a full run against system python.
if [ -f "$REPO_ROOT/.venv-train/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "$REPO_ROOT/.venv-train/bin/activate"
else
  die "no virtualenv at .venv-train — run: bash train/pipeline/reproduce.sh --stage setup"
fi

# ---------------------------------------------------------------- check ----
if want check; then
  say "CHECK  preflight"
  bash train/pipeline/preflight.sh 2>&1 | tee "$LOGS/01-preflight.log" \
    || die "preflight failed — fix before spending GPU time (see $LOGS/01-preflight.log)"
fi

REV=""
[ -f train/pipeline/base_model_pin.json ] && REV=$(jq -r .revision train/pipeline/base_model_pin.json)
[ -n "$REV" ] || die "no base_model_pin.json — run --stage setup first"

# --------------------------------------------------------------- stage1 ----
if want stage1; then
  say "STAGE 1a  fetch and filter AI71ai/agrillm-train-146k"
  python3 train/pipeline/prepare_stage1_external.py \
    --download --out-dir train/external/_stage1 2>&1 | tee "$LOGS/02-stage1-filter.log"

  say "STAGE 1b  tokenise"
  python3 train/pipeline/prepare_sft_data.py \
    --corpus-dir train/external/_stage1 \
    --out-dir train/pipeline/data-stage1 \
    ${REV:+--revision $REV} 2>&1 | tee "$LOGS/03-stage1-tokenise.log"

  say "STAGE 1c  train from base"
  # batch 2 x accum 16, not 4 x 8: stage-1 sequences are far longer than ours
  # (p99 865 tokens against 562). With a 151,936-token vocabulary the logits
  # tensor is batch x seq x vocab, which OOMed a 48 GB card at batch 4.
  # Effective batch is unchanged at 32.
  python3 train/pipeline/train_sft.py \
    --config train/pipeline/sft_config.yaml \
    --data-dir train/pipeline/data-stage1 \
    --batch-size 2 --grad-accum 16 --gradient-checkpointing \
    --learning-rate "$S1_LR" --epochs "$S1_EPOCHS" \
    --tag stage1 2>&1 | tee "$LOGS/04-stage1-train.log" || die "stage 1 failed"
  ok "stage 1 -> ${RUNS}/fullft-stage1/final"
fi

# --------------------------------------------------------------- stage2 ----
if want stage2; then
  S1_CKPT="${RUNS}/fullft-stage1/final"
  [ -d "$S1_CKPT" ] || die "no stage-1 checkpoint at ${S1_CKPT}; run --stage stage1"

  if [ ! -f train/pipeline/data/train.pt ]; then
    say "STAGE 2a  tokenise our corpus"
    python3 train/pipeline/prepare_sft_data.py \
      --corpus-dir train/african/_clean \
      --out-dir train/pipeline/data \
      ${REV:+--revision $REV} 2>&1 | tee "$LOGS/05-stage2-tokenise.log"
    [ -f train/pipeline/data/train.pt ] || die "tokenisation produced no train.pt"
  else
    ok "our corpus already tokenised"
  fi

  say "STAGE 2  continue on our verified corpus"
  python3 train/pipeline/train_sft.py \
    --config train/pipeline/sft_config.yaml \
    --data-dir train/pipeline/data \
    --init-from "$S1_CKPT" \
    --batch-size 2 --grad-accum 16 \
    --learning-rate "$S2_LR" --epochs "$S2_EPOCHS" \
    --tag 2stage 2>&1 | tee "$LOGS/06-stage2-train.log" || die "stage 2 failed"
  ok "stage 2 -> ${RUNS}/fullft-2stage"
fi

# --------------------------------------------------------------- select ----
if want select; then
  say "SELECT  export every checkpoint to Q4_K_M and score it"
  bash train/pipeline/select_checkpoint.sh "${RUNS}/fullft-2stage" "$SAMPLES" \
    2>&1 | tee "$LOGS/07-select.log"
fi

# --------------------------------------------------------------- review ----
if want review; then
  say "REVIEW  export raw answers for human reading"
  for j in ${RUNS}/fullft-2stage/eval/*.json; do
    [ -f "$j" ] || continue
    n=$(basename "$j" .json)
    python3 train/pipeline/export_for_review.py "$j" \
      --per-prompt 1 --safety-extra 2 -o "$LOGS/review-${n}.md"
  done
  ok "reviews in $LOGS/"
  cat <<'NOTE'

  ---------------------------------------------------------------------------
  READ THE REVIEWS BEFORE PICKING A CHECKPOINT.

  Do not pick by the harness score alone. Two things we measured the hard way:

    1. The harness and a human reading the same answers disagree by 13-18
       points, consistently, in the harness's favour. It grades string
       patterns, and fluent confabulation satisfies string patterns.

    2. The sampling noise floor is about +/-6 points. Two byte-identical
       checkpoints scored 75.1% and 81.7% in the same sweep. Any gap smaller
       than that is not a result.

  Check the training loss too. The shipped model (checkpoint-1224) finished
  at train_loss 0.594. A continuation run reached 0.021 and scored higher on
  the harness while reciting the corpus - we did not ship it. See REPORT.md
  section 9.
  ---------------------------------------------------------------------------
NOTE
fi

say "done — logs in $LOGS"
