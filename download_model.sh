#!/usr/bin/env bash
# ADTC 2026 — model download script.
#
# Contract (from the submission template):
#   - idempotent: safe to run repeatedly, must not re-download
#   - no credentials: weights must be publicly readable
#   - output path must exactly match _runtime.model_path in metadata.json
#
# This runs BEFORE the profiler starts. Once profiling begins, no outbound
# network requests are permitted.

set -euo pipefail

# --- keep these three in sync with metadata.json ------------------------------
MODEL_PATH="model/adtc-agri.gguf"
MODEL_URL="https://huggingface.co/shemking/agrillm-qwen2.5-1.5b-agri/resolve/main/adtc-agri-Q4_K_M.gguf"
EXPECTED_SHA256="c35e00652cd1e33638fc78c1289076bcd091ebc5bfc45463388a8223b47bd5aa"
# ------------------------------------------------------------------------------

cd "$(dirname "$0")"
mkdir -p "$(dirname "$MODEL_PATH")"

verify() {
  # Returns 0 if the file on disk matches the expected digest.
  [ -f "$MODEL_PATH" ] || return 1
  [ "$EXPECTED_SHA256" != "TODO-sha256-of-your-gguf" ] || return 0  # not yet pinned
  if command -v sha256sum >/dev/null 2>&1; then
    echo "${EXPECTED_SHA256}  ${MODEL_PATH}" | sha256sum -c --status
  elif command -v shasum >/dev/null 2>&1; then
    echo "${EXPECTED_SHA256}  ${MODEL_PATH}" | shasum -a 256 -c --status
  else
    echo "warning: no sha256 tool available, skipping integrity check" >&2
    return 0
  fi
}

if verify; then
  echo "✓ ${MODEL_PATH} already present and verified — nothing to do."
  exit 0
fi

if [ -f "$MODEL_PATH" ]; then
  echo "! ${MODEL_PATH} present but failed checksum — re-downloading." >&2
  rm -f "$MODEL_PATH"
fi

echo "→ downloading model to ${MODEL_PATH}"
# --location follows HF redirects to the CDN; --fail turns HTTP errors into
# a non-zero exit instead of a valid-looking HTML error page on disk.
curl --fail --location --retry 3 --retry-delay 5 \
     --output "${MODEL_PATH}.partial" \
     "$MODEL_URL"

mv "${MODEL_PATH}.partial" "$MODEL_PATH"

if ! verify; then
  echo "✗ checksum mismatch after download" >&2
  exit 1
fi

# Cheap sanity check: every GGUF file begins with the magic bytes "GGUF".
magic=$(head -c 4 "$MODEL_PATH")
if [ "$magic" != "GGUF" ]; then
  echo "✗ ${MODEL_PATH} is not a GGUF file (magic bytes: ${magic})" >&2
  exit 1
fi

echo "✓ model ready at ${MODEL_PATH} ($(du -h "$MODEL_PATH" | cut -f1))"
