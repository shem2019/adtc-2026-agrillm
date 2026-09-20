#!/usr/bin/env bash
# End-to-end driver for the AgriLLM full fine-tune on a rented H100.
#
#   source .venv-train/bin/activate
#   bash train/pipeline/run_pipeline.sh                 # everything
#   bash train/pipeline/run_pipeline.sh --stage data    # one stage only
#   bash train/pipeline/run_pipeline.sh --lrs 1e-5      # skip the sweep
#
# Stages: gates -> data -> train -> select -> quant -> verify -> provenance
#
# Expected wall clock on one H100 80GB, 6,703-row corpus:
#   gates       ~10 s
#   data        ~1 min
#   train       ~8-12 min per learning rate (4 epochs, ~840 steps)
#   select      ~5-8 min per checkpoint (export + quantise + 24 prompts x 8)
#   quant       ~25 min (4 precisions x 24 prompts x 10 samples)
#   verify      ~10-15 min (CPU inference, deliberately - grader conditions)
#   provenance  ~30 s
# Three learning rates x 4 checkpoints = 12 candidates, and the whole run lands
# around 3-4 hours, i.e. $12-16 of GPU time at $3.822/hr. That is deliberate:
# the corpus is the binding constraint on quality here, not compute, so the
# extra spend goes on reducing SELECTION variance and on measuring the
# quantisation trade rather than on training longer.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

STAGE="all"
# Tuned for impact rather than for cost. Training this model is minutes of GPU
# time, so the budget goes where it actually reduces risk:
#   - three learning rates, not two, bracketing the likely optimum
#   - four epochs with per-epoch checkpoints, giving 12 candidates in total
#   - 8 eval samples per prompt instead of 3, because at temperature 0.8 a
#     3-sample score is noisy enough to rank the wrong checkpoint first, and
#     picking the wrong checkpoint wastes everything upstream of it
LRS="5e-6 1e-5 2e-5"
EPOCHS=4
SAMPLES=8
CORPUS_DIR="train/african/_clean"
DATA_DIR="train/pipeline/data"
RUNS_DIR="train/pipeline/runs"
LLAMA_BIN="${REPO_ROOT}/.llama.cpp/build/bin"

while [ $# -gt 0 ]; do
  case "$1" in
    --stage)   STAGE="$2"; shift 2 ;;
    --lrs)     LRS="$2"; shift 2 ;;
    --epochs)  EPOCHS="$2"; shift 2 ;;
    --samples) SAMPLES="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 1 ;;
  esac
done

say()  { printf '\n\033[1;36m######## %s\033[0m\n' "$*"; }
ok()   { printf '\033[1;32m  ok  %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m  !!  %s\033[0m\n' "$*"; }
die()  { printf '\n\033[1;31mXX  %s\033[0m\n' "$*" >&2; exit 1; }
want() { [ "$STAGE" = "all" ] || [ "$STAGE" = "$1" ]; }

# ============================================================ stage: gates
# Interlock. The corpus has been silently poisoned once already in this
# project's history - by a generator that produced 22,300 rows from ~700
# underlying answers wrapped in serial numbers, which passed a naive
# distinctness check at 100%. verify_corpus.py is the check that catches that
# family of failure. If it does not pass, there is no point burning GPU time.
if want gates; then
  say "STAGE 1/7  corpus gates"
  python3 train/verify_corpus.py ${CORPUS_DIR}/*.jsonl | tail -5
  ROWS=$(cat ${CORPUS_DIR}/*.jsonl | wc -l | tr -d ' ')
  ok "corpus: ${ROWS} rows across $(ls ${CORPUS_DIR}/*.jsonl | wc -l | tr -d ' ') files"
fi

# ============================================================= stage: data
if want data; then
  say "STAGE 2/7  tokenise"
  PIN_ARGS=""
  if [ -f train/pipeline/base_model_pin.json ]; then
    REV=$(jq -r .revision train/pipeline/base_model_pin.json)
    PIN_ARGS="--revision ${REV}"
    ok "base model pinned at ${REV}"
  else
    warn "no base_model_pin.json - running unpinned. Gate 2 sec 3.1 wants the SHA."
  fi

  python3 train/pipeline/prepare_sft_data.py \
    --corpus-dir "$CORPUS_DIR" \
    --out-dir "$DATA_DIR" \
    ${PIN_ARGS}

  warn "Open ${DATA_DIR}/sample_rendered.txt and read two examples before training."
  warn "If the 'tokens loss is computed on' block holds anything but assistant"
  warn "text, stop: that misalignment is what produced the Round 1 chat drift."
fi

# ============================================================ stage: train
if want train; then
  say "STAGE 3/7  full fine-tune (sweep: ${LRS})"
  for LR in $LRS; do
    TAG="lr${LR}"
    say "  training ${TAG}"
    python3 train/pipeline/train_sft.py \
      --config train/pipeline/sft_config.yaml \
      --data-dir "$DATA_DIR" \
      --learning-rate "$LR" \
      --epochs "$EPOCHS" \
      --tag "$TAG" 2>&1 | tee "${RUNS_DIR}-${TAG}.log" || die "training failed for ${TAG}"
  done
fi

# =========================================================== stage: select
if want select; then
  say "STAGE 4/7  behavioural checkpoint selection"
  for LR in $LRS; do
    RUN="${RUNS_DIR}/fullft-lr${LR}"
    [ -d "$RUN" ] || { warn "no run dir ${RUN}, skipping"; continue; }
    bash train/pipeline/select_checkpoint.sh "$RUN" "$SAMPLES"
  done

  say "Cross-run comparison"
  python3 - "$RUNS_DIR" <<'PY'
import csv, pathlib, sys
runs = pathlib.Path(sys.argv[1])
rows = []
for summary in sorted(runs.glob("*/eval/summary.tsv")):
    run = summary.parent.parent.name
    for r in csv.DictReader(summary.open(), delimiter="\t"):
        if r["overall"] == "ERROR":
            continue
        rows.append((run, r["checkpoint"], float(r["overall"]),
                     int(r["safety_critical"]), r["gguf_mb"]))
if not rows:
    print("  no results yet"); raise SystemExit
rows.sort(key=lambda t: (t[3], -t[2]))
print(f"\n  {'run':20} {'checkpoint':18} {'overall':>8} {'crit':>5} {'MB':>5}")
print("  " + "-" * 62)
for run, ck, ov, crit, mb in rows:
    print(f"  {run:20} {ck:18} {ov:7.1%} {crit:5d} {mb:>5}")
clean = [r for r in rows if r[3] == 0]
print()
print(f"  BEST: {clean[0][0]}/{clean[0][1]} at {clean[0][2]:.1%}" if clean
      else "  NO CLEAN CANDIDATE - do not ship; see notes in select_checkpoint.sh")
PY
fi

# ============================================================ stage: quant
# Measure what 4-bit quantisation costs this model in accuracy. Round 1 shipped
# Q4_K_M purely for size and never checked. Accuracy is weighted 0.50 against
# efficiency's 0.20, so a larger file that answers better may well be the better
# trade - but only if the gain is measured rather than assumed.
if want quant; then
  say "STAGE 5/7  quantisation tier comparison"
  CAND_CKPT="${CANDIDATE_CKPT:-}"
  if [ -z "$CAND_CKPT" ]; then
    warn "CANDIDATE_CKPT not set; skipping. Re-run after picking a winner:"
    warn "  CANDIDATE_CKPT=${RUNS_DIR}/fullft-lr1e-5/checkpoint-XXX \\"
    warn "    bash train/pipeline/run_pipeline.sh --stage quant"
  else
    bash train/pipeline/compare_quants.sh "$CAND_CKPT" 10
  fi
fi

# =========================================================== stage: verify
# The selection loop ran with GPU offload for speed. The graders run llama.cpp
# on CPU with stock sampling. This stage re-runs the chosen candidate under
# those conditions, because that is the number that will actually be scored.
if want verify; then
  say "STAGE 6/7  CPU verification under grader conditions"
  CAND="${CANDIDATE_GGUF:-}"
  if [ -z "$CAND" ]; then
    CAND=$(ls -1 ${RUNS_DIR}/*/gguf/*-Q4_K_M.gguf 2>/dev/null | head -1 || true)
    [ -n "$CAND" ] && warn "CANDIDATE_GGUF not set; defaulting to ${CAND}"
  fi
  [ -n "$CAND" ] && [ -f "$CAND" ] || die "set CANDIDATE_GGUF=<path to winning .gguf> and re-run --stage verify"

  ok "verifying ${CAND}"
  "${LLAMA_BIN}/llama-server" -m "$CAND" -ngl 0 -c 4096 --port 8098 \
      > train/pipeline/verify-server.log 2>&1 &
  SPID=$!
  trap 'kill $SPID 2>/dev/null || true' EXIT
  # /health returns 503 while the model loads; curl -s exits 0 on that too, so
  # wait on the status code. CPU inference makes this load slow - allow longer.
  for _ in $(seq 1 300); do
    [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8098/health 2>/dev/null || echo 000)" = "200" ] && break
    sleep 1
  done

  # run_eval.py exits 1 when there is a safety-critical failure, which is
  # precisely the case we need to survive long enough to report. Without this
  # set +e, `set -euo pipefail` would kill the script right here and the
  # operator would see a bare non-zero exit instead of the finding.
  set +e
  python3 eval/run_eval.py --url http://127.0.0.1:8098 --profile grader \
      --samples 5 --out train/pipeline/verify-grader.json | tee train/pipeline/verify-grader.txt
  EVAL_RC=${PIPESTATUS[0]}
  set -e

  kill $SPID 2>/dev/null || true; trap - EXIT
  [ $EVAL_RC -eq 0 ] && ok "no safety-critical failures on CPU at 5 samples" \
                     || warn "safety-critical failure(s) under grader conditions - do not ship"
fi

# ======================================================= stage: provenance
if want provenance; then
  say "STAGE 7/7  provenance capture (Gate 2 section 3.1)"
  python3 train/pipeline/capture_provenance.py \
    --runs-dir "$RUNS_DIR" \
    --data-dir "$DATA_DIR" \
    --corpus-dir "$CORPUS_DIR" \
    --out-dir provenance
fi

say "done"
