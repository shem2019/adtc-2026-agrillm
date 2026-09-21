#!/usr/bin/env bash
# Score every checkpoint of a run with eval/run_eval.py and rank them.
#
#   bash train/pipeline/select_checkpoint.sh train/pipeline/runs/fullft [samples]
#
# Why this exists: validation loss cannot see that "induce vomiting" is a
# disqualifying answer, that a species name was invented, or that a reply ran 90
# words past a stated limit. Those are the three things that actually lost
# Round 1, and eval/run_eval.py checks all of them with string rules. So the
# winning checkpoint is chosen behaviourally, and loss is only ever a sanity
# signal.
#
# Each checkpoint is exported to Q4_K_M and served through llama.cpp, because
# that is the artefact the graders receive - not the bf16 HF weights. A
# checkpoint that looks fine in torch and degrades under 4-bit quantisation is
# a checkpoint we want to find here rather than after submission.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

RUN_DIR="${1:?usage: select_checkpoint.sh <run_dir> [samples]}"
SAMPLES="${2:-3}"

LLAMA_BIN="${REPO_ROOT}/.llama.cpp/build/bin"
GGUF_DIR="${RUN_DIR}/gguf"
EVAL_DIR="${RUN_DIR}/eval"
PORT=8099

say()  { printf '\n\033[1;32m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m!!  %s\033[0m\n' "$*"; }
die()  { printf '\n\033[1;31mXX  %s\033[0m\n' "$*" >&2; exit 1; }

[ -d "$RUN_DIR" ] || die "run dir not found: $RUN_DIR"
mkdir -p "$GGUF_DIR" "$EVAL_DIR"

SERVER_PID=""
cleanup() { [ -n "$SERVER_PID" ] && kill "$SERVER_PID" 2>/dev/null || true; }
trap cleanup EXIT

# llama.cpp's /health returns 503 with {"status":"loading model"} while the
# model is still loading, and 200 only once it can actually serve. `curl -s`
# exits 0 on ANY completed HTTP transaction including that 503, so checking
# curl's exit status returns the moment the socket is listening and the eval
# then connects to a server that is not ready. Check the status code.
wait_for_server() {
  local code
  for _ in $(seq 1 180); do
    code=$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:${PORT}/health" 2>/dev/null || echo 000)
    [ "$code" = "200" ] && return 0
    sleep 1
  done
  echo "      last /health status: ${code:-none}" >&2
  return 1
}

# Collect checkpoints in training order, final last.
# Sort on the numeric step suffix only: sorting whole paths on "-" breaks
# because the run dir itself contains dashes (fullft-lr1e-5).
mapfile -t CKPTS < <(find "$RUN_DIR" -maxdepth 1 -type d -name 'checkpoint-*' -print0 \
                     | xargs -0 -r -n1 basename \
                     | sort -t- -k2 -n \
                     | sed "s|^|${RUN_DIR%/}/|")
[ -d "${RUN_DIR}/final" ] && CKPTS+=("${RUN_DIR}/final")
[ "${#CKPTS[@]}" -gt 0 ] || die "no checkpoints found under $RUN_DIR"

say "Found ${#CKPTS[@]} checkpoint(s) to score, ${SAMPLES} samples each"
printf '  %s\n' "${CKPTS[@]}"

SUMMARY="${EVAL_DIR}/summary.tsv"
printf 'checkpoint\toverall\tsafety_critical\tgguf_mb\n' > "$SUMMARY"

for CKPT in "${CKPTS[@]}"; do
  NAME="$(basename "$CKPT")"
  say "=== ${NAME} ==="

  BASE="${GGUF_DIR}/${NAME}"
  GGUF="${BASE}-Q4_K_M.gguf"

  if [ ! -f "$GGUF" ]; then
    bash train/pipeline/export_gguf.sh "$CKPT" "$BASE" Q4_K_M
  else
    echo "  (reusing existing $GGUF)"
  fi

  # -ngl 99 here: selection only needs the ranking, and GPU offload turns a
  # 20-minute sweep into a 2-minute one. The FINAL candidate gets re-verified
  # on CPU with -ngl 0 in run_pipeline.sh, which is how the graders run it.
  "${LLAMA_BIN}/llama-server" -m "$GGUF" -ngl 99 -c 4096 --port "$PORT" \
      > "${EVAL_DIR}/${NAME}-server.log" 2>&1 &
  SERVER_PID=$!

  if ! wait_for_server; then
    warn "server failed to start for ${NAME}; see ${EVAL_DIR}/${NAME}-server.log"
    kill "$SERVER_PID" 2>/dev/null || true; SERVER_PID=""
    continue
  fi

  set +e
  python3 eval/run_eval.py \
      --url "http://127.0.0.1:${PORT}" \
      --profile grader \
      --samples "$SAMPLES" \
      --out "${EVAL_DIR}/${NAME}.json" \
      | tee "${EVAL_DIR}/${NAME}.txt"
  set -e

  kill "$SERVER_PID" 2>/dev/null || true
  wait "$SERVER_PID" 2>/dev/null || true
  SERVER_PID=""

  python3 - "$NAME" "${EVAL_DIR}/${NAME}.json" "$GGUF" "$SUMMARY" <<'PY'
import json, pathlib, sys
name, res_path, gguf, summary = sys.argv[1:5]
p = pathlib.Path(res_path)
if not p.exists():
    pathlib.Path(summary).open("a").write(f"{name}\tERROR\tERROR\t-\n"); raise SystemExit
d = json.loads(p.read_text())
crit = [r for r in d["results"] if r["category"] == "safety_critical" and r["score"] < 1]
mb = pathlib.Path(gguf).stat().st_size // 1024 // 1024
pathlib.Path(summary).open("a").write(
    f"{name}\t{d['overall']:.4f}\t{len(crit)}\t{mb}\n")
PY
done

say "Ranked results (safety-critical failures break the tie first)"
python3 - "$SUMMARY" <<'PY'
import csv, sys, pathlib
rows = list(csv.DictReader(pathlib.Path(sys.argv[1]).open(), delimiter="\t"))
ok = [r for r in rows if r["overall"] != "ERROR"]
for r in ok:
    r["overall_f"] = float(r["overall"])
    r["crit_i"] = int(r["safety_critical"])
ok.sort(key=lambda r: (r["crit_i"], -r["overall_f"]))

print(f"\n  {'checkpoint':22} {'overall':>8}  {'safety-crit fails':>17}  {'MB':>5}")
print("  " + "-" * 58)
for r in ok:
    flag = "" if r["crit_i"] == 0 else "  <-- disqualifying"
    print(f"  {r['checkpoint']:22} {r['overall_f']:7.1%}  {r['crit_i']:17d}  {r['gguf_mb']:>5}{flag}")

clean = [r for r in ok if r["crit_i"] == 0]
errored = [r for r in rows if r["overall"] == "ERROR"]
print()
if not ok:
    # Distinguish "every model failed safety" from "nothing was measured at all".
    # These need completely different responses and must never be conflated.
    print(f"  NO RESULTS AT ALL - {len(errored)} checkpoint(s) errored, 0 scored.")
    print("  This says nothing about the models; the eval never ran. Check the")
    print("  server logs next to this file for why llama-server did not come up:")
    print(f"    {sys.argv[1].rsplit('/', 1)[0] if '/' in sys.argv[1] else '.'}/*-server.log")
elif clean:
    w = clean[0]
    print(f"  WINNER: {w['checkpoint']}  ({w['overall_f']:.1%} overall, 0 safety-critical failures)")
    print(f"  Verify it on CPU before shipping - that is how the graders run it.")
    if errored:
        print(f"  NOTE: {len(errored)} checkpoint(s) errored and were not scored.")
else:
    print(f"  NO CLEAN CHECKPOINT among the {len(ok)} that scored.")
    print("  Every one has at least one safety-critical failure. Do not ship these.")
    print("  Lower the learning rate, or check the loss mask in")
    print("  train/pipeline/data/sample_rendered.txt, and retrain.")
PY

echo
echo "per-checkpoint detail: ${EVAL_DIR}/"
