#!/usr/bin/env bash
# Bootstrap a freshly rented H100 box for the AgriLLM full fine-tune.
# Idempotent: safe to re-run if the box reboots or a step fails halfway.
#
#   git clone https://github.com/shem2019/adtc-2026-agrillm.git
#   cd adtc-2026-agrillm
#   bash train/pipeline/setup_gpu.sh
#
# Takes roughly 10-15 minutes on a clean box, most of it llama.cpp compiling
# and the base model downloading.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

VENV="${REPO_ROOT}/.venv-train"
LLAMA_DIR="${REPO_ROOT}/.llama.cpp"
BASE_MODEL="Qwen/Qwen2.5-1.5B-Instruct"

say() { printf '\n\033[1;32m==> %s\033[0m\n' "$*"; }
warn() { printf '\n\033[1;33m!!  %s\033[0m\n' "$*"; }
die() { printf '\n\033[1;31mXX  %s\033[0m\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------- 1. the GPU
say "Checking GPU"
command -v nvidia-smi >/dev/null || die "nvidia-smi not found - is this actually a GPU box?"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

# Full fine-tune of Qwen2.5-1.5B peaks around 35GB at batch_size 4: ~23GB static
# (fp32 weights + gradients + AdamW states) plus logits and activations. 40GB is
# the practical floor, 48GB is comfortable. An H100 80GB is roughly twice what
# this needs - measured, not assumed.
VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | tr -d ' ')
if [ "${VRAM_MB}" -lt 38000 ]; then
  warn "Card reports ${VRAM_MB} MiB. Full FT peaks near 35GB at batch_size 4."
  warn "Set batch_size 2 / grad_accum 16 in sft_config.yaml (same effective batch),"
  warn "or use an 8-bit optimiser. Below ~24GB, full fine-tuning is not viable."
elif [ "${VRAM_MB}" -lt 46000 ]; then
  say "VRAM ${VRAM_MB} MiB - workable, limited headroom above the ~35GB peak"
else
  say "VRAM OK: ${VRAM_MB} MiB (peak need ~35GB)"
fi

# --------------------------------------------------------- 2. system packages
say "Installing system packages"
export DEBIAN_FRONTEND=noninteractive
sudo apt-get update -qq
sudo apt-get install -y -qq \
  build-essential cmake git git-lfs curl jq python3-venv python3-pip ccache
git lfs install --skip-repo || true

# build-essential has twice now failed to leave a working C++ compiler on these
# rented images. The symptom is not obvious: gcc exists, nvcc exists, and the
# llama.cpp CUDA build dies much later with
#   gcc: fatal error: cannot execute 'cc1plus': execvp: No such file or directory
# because cc1plus is the C++ backend and ships with g++, not gcc. Check for the
# binary itself rather than trusting the package install to have worked.
if ! command -v g++ >/dev/null 2>&1 || ! ls /usr/lib/gcc/x86_64-linux-gnu/*/cc1plus >/dev/null 2>&1; then
  warn "g++ or cc1plus missing after build-essential; reinstalling explicitly"
  sudo apt-get install -y -qq --reinstall build-essential g++ gcc
fi
command -v g++ >/dev/null 2>&1 || die "no working g++ - llama.cpp cannot build with CUDA"
say "C++ toolchain: $(g++ --version | head -1)"

# ------------------------------------------------------------- 3. python env
if [ ! -d "$VENV" ]; then
  say "Creating virtualenv at $VENV"
  python3 -m venv "$VENV"
fi
# shellcheck disable=SC1091
source "${VENV}/bin/activate"
python -m pip install --quiet --upgrade pip wheel setuptools

say "Installing PyTorch (CUDA build)"
# Default PyPI wheels on linux ship bundled CUDA 12.x and work on H100 (sm_90).
# If this box has an unusual driver, swap to an explicit index, e.g.:
#   pip install torch --index-url https://download.pytorch.org/whl/cu124
python -m pip install --quiet torch

say "Installing training stack"
python -m pip install --quiet \
  "transformers>=4.45" \
  "accelerate>=0.34" \
  "huggingface_hub>=0.25" \
  safetensors sentencepiece protobuf pyyaml numpy

say "Verifying torch sees the GPU"
python - <<'PY'
import torch, sys
if not torch.cuda.is_available():
    sys.exit("torch cannot see a CUDA device")
p = torch.cuda.get_device_properties(0)
print(f"  torch {torch.__version__}  CUDA {torch.version.cuda}")
print(f"  device: {p.name}  {p.total_memory/1024**3:.1f} GB  sm_{p.major}{p.minor}")
x = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
print(f"  bf16 matmul check: {(x @ x).float().mean().item():.4f}")
PY

# --------------------------------------------------------------- 4. llama.cpp
if [ ! -d "$LLAMA_DIR" ]; then
  say "Cloning llama.cpp"
  git clone --depth 1 https://github.com/ggml-org/llama.cpp "$LLAMA_DIR"
fi

if [ ! -x "${LLAMA_DIR}/build/bin/llama-quantize" ]; then
  say "Building llama.cpp with CUDA (used for fast checkpoint eval)"
  cmake -S "$LLAMA_DIR" -B "${LLAMA_DIR}/build" \
    -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON >/dev/null
  cmake --build "${LLAMA_DIR}/build" --config Release -j"$(nproc)" \
    --target llama-quantize llama-server llama-cli >/dev/null
fi
say "llama.cpp binaries"
ls -1 "${LLAMA_DIR}/build/bin/" | grep -E '^llama-(quantize|server|cli)$' || die "llama.cpp build incomplete"

say "Installing llama.cpp conversion requirements (excluding torch)"
# llama.cpp's convert requirements pin torch AND point at PyTorch's CPU wheel
# index, because conversion only needs to read tensors and a CPU build is
# smaller. Installing them unfiltered silently replaces the CUDA torch above
# with a +cpu build, and the failure surfaces much later as
# "CUDA not available" at the start of training. So strip torch and any index
# directives out, and install the rest.
REQ_SRC=""
for cand in "${LLAMA_DIR}/requirements/requirements-convert_hf_to_gguf.txt" \
            "${LLAMA_DIR}/requirements.txt"; do
  [ -f "$cand" ] && REQ_SRC="$cand" && break
done

if [ -n "$REQ_SRC" ]; then
  REQ_FILTERED="$(mktemp)"
  grep -viE '^\s*(torch([=<>~!].*)?|--extra-index-url.*|--index-url.*)\s*$' \
    "$REQ_SRC" > "$REQ_FILTERED" || true
  python -m pip install --quiet -r "$REQ_FILTERED" || \
    warn "Some llama.cpp convert requirements failed - check before export"
  rm -f "$REQ_FILTERED"
else
  warn "Could not find llama.cpp convert requirements - check before export"
fi

# Hard guard: if anything above downgraded torch, fail here rather than at the
# start of a training run.
say "Re-verifying torch still has CUDA after dependency installs"
python - <<'PY'
import sys, torch
ok = torch.cuda.is_available()
print(f"  torch {torch.__version__}  cuda_available={ok}")
if not ok or torch.version.cuda is None:
    sys.exit(
        "torch lost CUDA support during dependency installation.\n"
        "  Fix with:  pip uninstall -y torch && pip install torch\n"
        "  Then re-run this script."
    )
PY

# -------------------------------------------------- 5. base model + commit SHA
say "Fetching base model and pinning its commit SHA"
python - "$BASE_MODEL" <<'PY'
import json, sys, pathlib
from huggingface_hub import HfApi, snapshot_download

repo = sys.argv[1]
sha = HfApi().model_info(repo).sha
print(f"  {repo} @ {sha}")
snapshot_download(repo, revision=sha)
out = pathlib.Path("train/pipeline/base_model_pin.json")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps({"base_model": repo, "revision": sha}, indent=2), encoding="utf-8")
print(f"  pinned -> {out}")
PY

PINNED_SHA=$(jq -r .revision train/pipeline/base_model_pin.json)

# ------------------------------------------------------------------ 6. wrap up
cat <<EOF

$(say "Setup complete")

  venv        : ${VENV}
  llama.cpp   : ${LLAMA_DIR}/build/bin
  base model  : ${BASE_MODEL}
  commit SHA  : ${PINNED_SHA}

Two things to do before the real run:

  1. Put that SHA into train/pipeline/sft_config.yaml as base_model_revision,
     and into metadata.json (Gate 2 section 3.1 asks for it):

       sed -i 's|^base_model_revision: null|base_model_revision: ${PINNED_SHA}|' \\
         train/pipeline/sft_config.yaml

  2. Activate the venv in every later shell:

       source ${VENV}/bin/activate

Then drive everything with:

       bash train/pipeline/run_pipeline.sh

EOF
