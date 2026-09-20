#!/usr/bin/env bash
# Measure what quantisation actually costs the winning checkpoint in accuracy.
#
#   bash train/pipeline/compare_quants.sh <checkpoint_dir> [samples]
#
#   bash train/pipeline/compare_quants.sh \
#     train/pipeline/runs/fullft-lr1e-5/checkpoint-406 10
#
# Why bother. The Round 1 submission shipped Q4_K_M at 986 MB and nobody ever
# measured what the 4-bit squeeze cost in answer quality - it was chosen for
# size and never checked against accuracy. But the ADTC formula weights accuracy
# at 0.50 and efficiency at only 0.20, so trading some file size back for
# correctness can be net positive. This exports the same checkpoint at four
# precisions and scores each one, so that trade is made with numbers instead of
# assumption.
#
# Rough sizes for a 1.54B model (all comfortably inside an 8 GB laptop):
#   Q4_K_M  ~1.0 GB     Q5_K_M  ~1.1 GB
#   Q6_K    ~1.3 GB     f16     ~3.1 GB  (ceiling reference, not shippable)
#
# f16 is included as the no-quantisation-loss ceiling. If Q4_K_M already matches
# it, quantisation is free and you ship the small one. If there is a real gap,
# the middle tiers tell you where it closes.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

CKPT="${1:?usage: compare_quants.sh <checkpoint_dir> [samples]}"
SAMPLES="${2:-10}"

LLAMA_BIN="${REPO_ROOT}/.llama.cpp/build/bin"
OUT_DIR="${CKPT}/quant-compare"
PORT=8097

say()  { printf '\n\033[1;32m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m!!  %s\033[0m\n' "$*"; }
die()  { printf '\n\033[1;31mXX  %s\033[0m\n' "$*" >&2; exit 1; }

[ -d "$CKPT" ] || die "checkpoint not found: $CKPT"
mkdir -p "$OUT_DIR"

SERVER_PID=""
cleanup() { [ -n "$SERVER_PID" ] && kill "$SERVER_PID" 2>/dev/null || true; }
trap cleanup EXIT

SUMMARY="${OUT_DIR}/summary.tsv"
printf 'quant\toverall\tsafety_critical\tsize_mb\n' > "$SUMMARY"

# Build f16 once; every quantised tier is derived from it.
F16="${OUT_DIR}/model-f16.gguf"
if [ ! -f "$F16" ]; then
  say "Exporting f16 base for quantisation"
  bash train/pipeline/export_gguf.sh "$CKPT" "${OUT_DIR}/model" Q4_K_M
fi

for QUANT in Q4_K_M Q5_K_M Q6_K f16; do
  say "=== ${QUANT} ==="
  GGUF="${OUT_DIR}/model-${QUANT}.gguf"

  if [ ! -f "$GGUF" ]; then
    if [ "$QUANT" = "f16" ]; then
      cp "$F16" "$GGUF" 2>/dev/null || true
    else
      "${LLAMA_BIN}/llama-quantize" "$F16" "$GGUF" "$QUANT" || {
        warn "quantise failed for ${QUANT}, skipping"; continue; }
    fi
  fi
  [ -f "$GGUF" ] || { warn "no file for ${QUANT}, skipping"; continue; }

  SIZE_MB=$(( $(stat -c%s "$GGUF" 2>/dev/null || stat -f%z "$GGUF") / 1024 / 1024 ))

  "${LLAMA_BIN}/llama-server" -m "$GGUF" -ngl 99 -c 4096 --port "$PORT" \
      > "${OUT_DIR}/${QUANT}-server.log" 2>&1 &
  SERVER_PID=$!
  # /health returns 503 while loading; curl -s exits 0 on that too, so check the
  # status code rather than curl's exit status.
  for _ in $(seq 1 180); do
    [ "$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:${PORT}/health" 2>/dev/null || echo 000)" = "200" ] && break
    sleep 1
  done

  set +e
  python3 eval/run_eval.py --url "http://127.0.0.1:${PORT}" --profile grader \
      --samples "$SAMPLES" --out "${OUT_DIR}/${QUANT}.json" \
      | tee "${OUT_DIR}/${QUANT}.txt" | tail -12
  set -e

  kill "$SERVER_PID" 2>/dev/null || true; wait "$SERVER_PID" 2>/dev/null || true
  SERVER_PID=""

  python3 - "$QUANT" "${OUT_DIR}/${QUANT}.json" "$SIZE_MB" "$SUMMARY" <<'PY'
import json, pathlib, sys
quant, res, size_mb, summary = sys.argv[1:5]
p = pathlib.Path(res)
if not p.exists():
    pathlib.Path(summary).open("a").write(f"{quant}\tERROR\tERROR\t{size_mb}\n"); raise SystemExit
d = json.loads(p.read_text())
crit = sum(1 for r in d["results"] if r["category"] == "safety_critical" and r["score"] < 1)
pathlib.Path(summary).open("a").write(f"{quant}\t{d['overall']:.4f}\t{crit}\t{size_mb}\n")
PY
done

say "Quantisation vs accuracy"
python3 - "$SUMMARY" "$SAMPLES" <<'PY'
import csv, pathlib, sys
rows = [r for r in csv.DictReader(pathlib.Path(sys.argv[1]).open(), delimiter="\t")
        if r["overall"] != "ERROR"]
samples = sys.argv[2]
if not rows:
    print("  no results"); raise SystemExit

order = {"Q4_K_M": 0, "Q5_K_M": 1, "Q6_K": 2, "f16": 3}
rows.sort(key=lambda r: order.get(r["quant"], 9))
ceiling = next((float(r["overall"]) for r in rows if r["quant"] == "f16"), None)

print(f"\n  scored at {samples} samples per prompt\n")
print(f"  {'quant':8} {'size':>7}  {'overall':>8}  {'vs f16':>8}  {'safety-crit':>11}")
print("  " + "-" * 52)
for r in rows:
    ov = float(r["overall"])
    delta = f"{ov - ceiling:+.1%}" if ceiling is not None and r["quant"] != "f16" else "  --"
    print(f"  {r['quant']:8} {r['size_mb']+' MB':>7}  {ov:7.1%}  {delta:>8}  {r['safety_critical']:>11}")

print()
shippable = [r for r in rows if r["quant"] != "f16" and int(r["safety_critical"]) == 0]
if not shippable:
    print("  No quantised tier is clean on safety-critical items. Do not ship.")
elif ceiling is None:
    print("  No f16 reference; compare tiers against each other.")
else:
    best = max(shippable, key=lambda r: float(r["overall"]))
    q4 = next((r for r in shippable if r["quant"] == "Q4_K_M"), None)
    print(f"  Best shippable tier: {best['quant']} at {float(best['overall']):.1%} ({best['size_mb']} MB)")
    if q4 and best["quant"] != "Q4_K_M":
        gain = float(best["overall"]) - float(q4["overall"])
        extra = int(best["size_mb"]) - int(q4["size_mb"])
        print(f"  Moving off Q4_K_M buys {gain:+.1%} accuracy for {extra:+d} MB.")
        print(f"  Accuracy is weighted 0.50 and efficiency 0.20, so if that gain is")
        print(f"  real and not sampling noise, the larger file is probably worth it.")
        print(f"  Re-run with more --samples if the gap is under about 3 points.")
    elif q4:
        print("  Q4_K_M is already the best shippable tier - quantisation is costing")
        print("  you nothing measurable. Ship the small one.")
PY

echo
echo "detail: ${OUT_DIR}/"
