#!/usr/bin/env bash
# Benchmark every candidate in candidates.tsv under audit-like constraints.
#
#   ./sweep.sh              # throughput + memory only (fast, ~2 min/model)
#   ./sweep.sh --accuracy   # also run lm-eval arc_easy (slow, ~20-40 min/model)
#
# Results land in bench/results/<label>.json — the same schema the real audit
# produces. Rank them with:  python3 bench/score.py

set -euo pipefail

cd "$(dirname "$0")"

IMAGE="adtc-bench:latest"
MODELS_DIR="$PWD/models"
RESULTS_DIR="$PWD/results"
STAGE_DIR="$PWD/.stage"

# Match the ADTC Standard Laptop profile. --memory=7.5g is what the profiler's
# own Dockerfile documents; 4 vCPU is the published reference.
CPUS="${ADTC_CPUS:-4}"
MEMORY="${ADTC_MEMORY:-7.5g}"

RUN_ACCURACY=0
FORCE=0
for arg in "$@"; do
  case "$arg" in
    --accuracy) RUN_ACCURACY=1 ;;
    --force)    FORCE=1 ;;
    -h|--help)
      sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'
      exit 0 ;;
    *) echo "unknown option: $arg (try --help)"; exit 2 ;;
  esac
done

mkdir -p "$MODELS_DIR" "$RESULTS_DIR" "$STAGE_DIR"

# lm-eval downloads its question set from HuggingFace on first use. Each model
# is profiled in a --rm container, so without a persistent cache that download
# repeats for every candidate — and a single network blip at 3am silently costs
# you one model's accuracy score. Mounting a host directory makes it a one-time
# fetch shared by every run.
HF_CACHE="$PWD/.hfcache"
mkdir -p "$HF_CACHE"

# Append every run to a log. A sweep that takes hours and dies at 3am is only
# useful if you can read back what it did — and REPORT.md needs the audit trail.
LOG="$RESULTS_DIR/sweep.log"
exec > >(tee -a "$LOG") 2>&1
echo ""
echo "════ sweep started $(date -u '+%Y-%m-%dT%H:%M:%SZ') "\
     "accuracy=$RUN_ACCURACY force=$FORCE cpus=$CPUS mem=$MEMORY"

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then

  # Fetch llama.cpp source on the HOST, where the network is fast. Docker
  # Desktop's VM network on macOS measured ~20x slower than the host, so
  # anything downloaded inside the build is a liability.
  LLAMA_REF="${LLAMA_CPP_REF:-master}"
  TARBALL="$PWD/llama.cpp.tar.gz"
  if [ ! -s "$TARBALL" ]; then
    echo "→ fetching llama.cpp source ($LLAMA_REF) on the host"
    url="https://codeload.github.com/ggml-org/llama.cpp/tar.gz/refs/heads/${LLAMA_REF}"
    if command -v aria2c >/dev/null 2>&1; then
      aria2c -x 8 -s 8 --continue=true --max-tries=8 --retry-wait=5 \
             --console-log-level=warn \
             -d "$PWD" -o "llama.cpp.tar.gz.partial" "$url" \
        && mv "$PWD/llama.cpp.tar.gz.partial" "$TARBALL"
    else
      curl -fL --retry 8 --retry-delay 5 --retry-all-errors --connect-timeout 30 \
           --progress-bar -o "${TARBALL}.partial" "$url" \
        && mv "${TARBALL}.partial" "$TARBALL"
    fi
    # gzip -t verifies the archive is complete and uncorrupted. A truncated
    # tarball would otherwise fail deep inside the Docker build.
    if ! gzip -t "$TARBALL" 2>/dev/null; then
      echo "✗ llama.cpp source download is corrupt or incomplete."
      echo "  Delete $TARBALL and re-run ./sweep.sh"
      rm -f "$TARBALL"
      exit 1
    fi
    echo "  ✓ $(du -h "$TARBALL" | cut -f1) verified"
  else
    echo "→ using cached llama.cpp source ($(du -h "$TARBALL" | cut -f1))"
  fi

  echo "→ building $IMAGE (one-off, ~10-15 min of compiling — CPU-bound, not network)"
  docker build -t "$IMAGE" .
fi

# Record the exact toolchain identity. llama.cpp is tracked at master, which is a
# moving target: the build number and commit below are what makes your benchmark
# numbers reproducible, and REPORT.md's reproducibility section needs them.
ENVFILE="$RESULTS_DIR/environment.txt"
# Every probe below is guarded with `|| true`. The script runs under
# `set -euo pipefail`, so an unguarded non-zero exit here — llama-bench does not
# accept --version on every build — would abort the whole sweep before a single
# model was measured. Recording provenance must never be able to stop the run.
{
  echo "captured:       $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  echo "host:           $(uname -srm)"
  echo "docker:         $(docker --version 2>/dev/null || echo unknown)"
  echo "container cpus: $CPUS"
  echo "container mem:  $MEMORY"
  echo -n "llama-bench:    "
  { docker run --rm --entrypoint llama-bench "$IMAGE" --version 2>&1 \
      | head -2 | tr '\n' ' '; } || echo -n "version probe unsupported"
  echo ""
  echo -n "profiler:       "
  { docker run --rm --entrypoint pip "$IMAGE" show adtc-profiler 2>/dev/null \
      | awk -F': ' '/^Version/{print $2}'; } || echo -n "unknown"
  echo ""
} > "$ENVFILE" 2>&1 || true
echo "→ toolchain recorded in $ENVFILE"
cat "$ENVFILE" | sed 's/^/    /'

# Warm the dataset cache ONCE, before the loop, while you are still awake.
# Failing here is a clear message now; failing inside model 5 at 3am is a
# silently missing accuracy score you would not notice until morning.
if [ "$RUN_ACCURACY" = "1" ]; then
  if [ -z "$(ls -A "$HF_CACHE" 2>/dev/null)" ]; then
    echo "→ fetching the lm-eval question set once (a few MB, needs internet)"
    if docker run --rm \
        -v "$HF_CACHE:/root/.cache/huggingface" \
        -e HF_HOME=/root/.cache/huggingface \
        --entrypoint python3 "$IMAGE" -c \
        "from datasets import load_dataset; load_dataset('allenai/ai2_arc','ARC-Easy'); print('ok')"
    then
      echo "  ✓ dataset cached — the rest of the run needs no internet"
    else
      echo "✗ could not fetch the benchmark dataset. Check your connection and"
      echo "  re-run. Nothing else has been done yet, so nothing is lost."
      exit 1
    fi
  else
    echo "→ lm-eval dataset already cached — this run needs no internet"
  fi
fi

# Read the manifest, skipping comments and blank lines.
while IFS=$'\t' read -r label repo filename params; do
  case "$label" in ''|'#'*) continue ;; esac
  [ -n "${filename:-}" ] || continue

  gguf="$MODELS_DIR/$filename"
  url="https://huggingface.co/${repo}/resolve/main/${filename}"

  echo ""
  echo "══ $label  ($params, $filename)"

  # Skip work already done. A 7-model sweep with accuracy enabled runs for
  # hours; if it dies at model 6 you must not redo models 1-5. Re-running is
  # now safe and cheap. Use --force to deliberately re-measure everything.
  existing="$RESULTS_DIR/${label}.json"
  if [ -f "$existing" ] && [ "$FORCE" = "0" ]; then
    if [ "$RUN_ACCURACY" = "1" ] && grep -q '"accuracy": *\[\]' "$existing"; then
      echo "→ result exists but has no accuracy data — re-running with accuracy"
    else
      echo "✓ already profiled ($existing) — skipping. Use --force to redo."
      continue
    fi
  fi

  if [ ! -f "$gguf" ]; then
    echo "→ downloading from $repo"
    # A single TCP stream to HuggingFace's CDN is latency-bound over a long
    # round-trip, not bandwidth-bound: it commonly stalls at a fraction of line
    # rate. aria2c opens 16 parallel connections and typically recovers most of
    # the gap. Both paths resume, and the partial file is deliberately NOT
    # deleted on failure so re-running ./sweep.sh picks up where it stopped.
    dl_ok=0
    if command -v aria2c >/dev/null 2>&1; then
      aria2c \
        --max-connection-per-server=16 \
        --split=16 \
        --min-split-size=1M \
        --continue=true \
        --max-tries=8 --retry-wait=5 \
        --connect-timeout=30 \
        --summary-interval=10 \
        --console-log-level=warn \
        --dir="$MODELS_DIR" \
        --out="${filename}.partial" \
        "$url" && dl_ok=1
    else
      echo "  (install aria2 for parallel downloads: brew install aria2)"
      curl --fail --location \
           --continue-at - \
           --retry 8 --retry-delay 5 --retry-all-errors \
           --connect-timeout 30 \
           --progress-bar \
           -o "${gguf}.partial" "$url" && dl_ok=1
    fi

    if [ "$dl_ok" != "1" ]; then
      echo "✗ download incomplete for $label — partial kept at ${gguf}.partial"
      echo "  re-run ./sweep.sh to resume it."
      continue
    fi
    mv "${gguf}.partial" "$gguf"
  fi

  # A truncated download still produces a file. Catch it here rather than
  # letting llama-bench fail with an opaque error 20 minutes later.
  if [ "$(head -c 4 "$gguf")" != "GGUF" ]; then
    echo "✗ $label is not a valid GGUF (truncated or an HTML error page). Removing."
    rm -f "$gguf"
    continue
  fi

  # The profiler needs a submission directory, not a bare .gguf. Build a
  # throwaway one per candidate. parameters_estimate must match the GGUF
  # header or gguf.fraud_check flags it — so we feed the manifest value and
  # let the report's params_match field tell us if the manifest is wrong.
  stage="$STAGE_DIR/$label"
  rm -rf "$stage"; mkdir -p "$stage/model"
  ln "$gguf" "$stage/model/candidate.gguf" 2>/dev/null || cp "$gguf" "$stage/model/candidate.gguf"

  cat > "$stage/metadata.json" <<JSON
{
  "team_id": "sweep-${label}",
  "domain": "agriculture",
  "language_scope": ["en"],
  "african_alpha_claim": false,
  "budget_laptop_claim": true,
  "submitter": {
    "name": "Shem Kinyanjui Njuguna",
    "email": "shem.kinyanjui.njuguna@gmail.com",
    "github_handle": "shem2019"
  },
  "cross_disciplinary_pairing": {
    "discipline": "agricultural_extension",
    "load_bearing": true,
    "description": "Candidate sweep run."
  },
  "test_prompts": [
    { "prompt_id": "tp_001", "prompt": "Placeholder prompt for sweep run one." },
    { "prompt_id": "tp_002", "prompt": "Placeholder prompt for sweep run two." }
  ],
  "model": {
    "name": "${label}",
    "runtime": "llama.cpp",
    "quantization": "GGUF",
    "parameters_estimate": "${params}",
    "packaging": "binary_bundle"
  },
  "_runtime": { "model_path": "model/candidate.gguf" }
}
JSON

  accuracy_flag="--skip-accuracy"
  [ "$RUN_ACCURACY" = "1" ] && accuracy_flag=""

  docker run --rm \
    --cpus="$CPUS" --memory="$MEMORY" \
    -v "$stage:/submission:ro" \
    -v "$RESULTS_DIR:/results" \
    -v "$HF_CACHE:/root/.cache/huggingface" \
    -e HF_HOME=/root/.cache/huggingface \
    --entrypoint adtc-profiler \
    "$IMAGE" run \
      --submission /submission \
      --mode participant \
      --output "/results/${label}.json" \
      $accuracy_flag \
  || echo "✗ profiler failed for $label (OOM? unsupported arch?) — continuing"

done < candidates.tsv

echo ""
echo "── sweep complete. Ranking:"
python3 score.py
