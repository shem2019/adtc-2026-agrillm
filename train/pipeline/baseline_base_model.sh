#!/usr/bin/env bash
# Score the UNTRAINED base model on the same harness, for comparison.
#
#   bash train/pipeline/baseline_base_model.sh [samples]
#
# Without this number, a fine-tuned score is meaningless. 33% could be four
# times better than the base model or identical to it, and those call for
# opposite decisions. This has been outstanding since the start of the project
# and its absence is why the sweep results cannot currently be interpreted.
#
# Scores stock Qwen2.5-1.5B-Instruct at the pinned commit, converted and
# quantised exactly the way the fine-tunes were, so the only variable is the
# training.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

SAMPLES="${1:-8}"
LLAMA_BIN="${REPO_ROOT}/.llama.cpp/build/bin"
OUT_DIR="${REPO_ROOT}/train/pipeline/baseline"
PORT=8096

say() { printf '\n\033[1;32m==> %s\033[0m\n' "$*"; }
die() { printf '\n\033[1;31mXX  %s\033[0m\n' "$*" >&2; exit 1; }

mkdir -p "$OUT_DIR"

PIN="${REPO_ROOT}/train/pipeline/base_model_pin.json"
[ -f "$PIN" ] || die "no base_model_pin.json - run setup_gpu.sh first"
REPO=$(jq -r .base_model "$PIN")
REV=$(jq -r .revision "$PIN")
say "Baseline: ${REPO} @ ${REV}"

# Materialise the pinned base model as a plain HF dir we can convert.
SNAP="${OUT_DIR}/hf"
if [ ! -f "${SNAP}/config.json" ]; then
  say "Materialising the pinned base model"
  python - "$REPO" "$REV" "$SNAP" <<'PY'
import sys, pathlib
from transformers import AutoModelForCausalLM, AutoTokenizer
repo, rev, out = sys.argv[1], sys.argv[2], sys.argv[3]
pathlib.Path(out).mkdir(parents=True, exist_ok=True)
AutoModelForCausalLM.from_pretrained(repo, revision=rev).save_pretrained(out)
AutoTokenizer.from_pretrained(repo, revision=rev).save_pretrained(out)
print(f"  -> {out}")
PY
fi

# NOTE: deliberately NOT patching the chat template here. The baseline is stock
# Qwen answering as stock Qwen, which is the honest comparison point.
GGUF="${OUT_DIR}/base-Q4_K_M.gguf"
if [ ! -f "$GGUF" ]; then
  say "Converting and quantising the base model the same way as the fine-tunes"
  python "${REPO_ROOT}/.llama.cpp/convert_hf_to_gguf.py" "$SNAP" \
    --outfile "${OUT_DIR}/base-f16.gguf" --outtype f16
  "${LLAMA_BIN}/llama-quantize" "${OUT_DIR}/base-f16.gguf" "$GGUF" Q4_K_M
fi

say "Serving the base model"
"${LLAMA_BIN}/llama-server" -m "$GGUF" -ngl 99 -c 4096 --port "$PORT" \
    > "${OUT_DIR}/server.log" 2>&1 &
SPID=$!
trap 'kill $SPID 2>/dev/null || true' EXIT

for _ in $(seq 1 180); do
  [ "$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:${PORT}/health" 2>/dev/null || echo 000)" = "200" ] && break
  sleep 1
done

say "Scoring (${SAMPLES} samples per prompt, grader settings)"
set +e
python3 eval/run_eval.py --url "http://127.0.0.1:${PORT}" --profile grader \
    --samples "$SAMPLES" --out "${OUT_DIR}/baseline.json" | tee "${OUT_DIR}/baseline.txt"
set -e

kill $SPID 2>/dev/null || true; trap - EXIT

say "Baseline vs fine-tuned"
python3 - "${OUT_DIR}/baseline.json" "${REPO_ROOT}/train/pipeline/runs" <<'PY'
import csv, json, pathlib, sys
base = json.loads(pathlib.Path(sys.argv[1]).read_text())
base_overall = base["overall"]
base_crit = sum(1 for r in base["results"]
                if r["category"] == "safety_critical" and r["score"] < 1)
print(f"\n  UNTRAINED BASE MODEL : {base_overall:6.1%}   safety-critical failures: {base_crit}")

rows = []
for s in sorted(pathlib.Path(sys.argv[2]).glob("*/eval/summary.tsv")):
    for r in csv.DictReader(s.open(), delimiter="\t"):
        if r["overall"] != "ERROR":
            rows.append((s.parent.parent.name, r["checkpoint"],
                         float(r["overall"]), int(r["safety_critical"])))
if rows:
    rows.sort(key=lambda t: -t[2])
    best = rows[0]
    print(f"  BEST FINE-TUNE       : {best[2]:6.1%}   safety-critical failures: {best[3]}"
          f"   ({best[0]}/{best[1]})")
    delta = best[2] - base_overall
    print(f"  DELTA                : {delta:+6.1%}")
    print()
    if delta < 0.02:
        print("  Training is not buying anything measurable. Do not tune hyperparameters;")
        print("  find out why. Read actual generations with inspect_eval.py.")
    elif best[3] >= base_crit:
        print("  Overall improved but safety-critical failures did NOT. That is the")
        print("  category the corpus was rebuilt to fix, so check whether the harness")
        print("  rules match how the model now phrases refusals.")
    else:
        print("  Training helps on both axes. Now decide whether it helps ENOUGH.")
PY
