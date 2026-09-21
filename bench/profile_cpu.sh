#!/usr/bin/env bash
# Measure the shipped model with the official ADTC profiler on a clean
# Ubuntu 24.04 machine (or WSL2 Ubuntu 24.04 on Windows), CPU only.
#
#   curl -fsSL https://raw.githubusercontent.com/shem2019/adtc-2026-agrillm/main/bench/profile_cpu.sh -o profile_cpu.sh
#   bash profile_cpu.sh
#
# Safe to re-run: finished steps are skipped. Log: ~/agrillm-profile.log
# Results: ~/agrillm-profile-results.tgz
set -euo pipefail
exec > >(tee -a "$HOME/agrillm-profile.log") 2>&1

LLAMA_REF=b10175
PROFILER_SHA=7f117dde3d8f2a0b3d3f05948a7bfd4bf693e909
MODEL_SHA=ad7e079f7cfd307edc7629a35c906cb55ed41218ba14a93e48120d81952d3e0f
REPO="$HOME/adtc-2026-agrillm"
JOBS=$(( $(nproc) > 4 ? 4 : $(nproc) ))

echo "== 0. machine"
lscpu | grep -E "Model name|^CPU\(s\)" || true
free -h

echo "== 1. toolchain"
sudo apt-get update -qq
sudo apt-get install -y -qq git curl build-essential cmake python3 python3-pip python3-venv

echo "== 2. llama.cpp $LLAMA_REF"
[ -d "$HOME/llama.cpp" ] || git clone --depth 1 --branch "$LLAMA_REF" https://github.com/ggml-org/llama.cpp "$HOME/llama.cpp"
cmake -S "$HOME/llama.cpp" -B "$HOME/llama.cpp/build" -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF
cmake --build "$HOME/llama.cpp/build" --config Release -j"$JOBS" --target llama-bench llama-cli llama-server
export PATH="$HOME/llama.cpp/build/bin:$PATH"
llama-bench --help > /dev/null && echo "llama-bench OK"

echo "== 3. repo + weights"
[ -d "$REPO/.git" ] || git clone https://github.com/shem2019/adtc-2026-agrillm.git "$REPO"
cd "$REPO"
git pull --ff-only origin main
git log -1 --oneline
bash download_model.sh
sha256sum model/adtc-agri-Q4_K_M.gguf | tee sha.txt
grep -q "$MODEL_SHA" sha.txt || { echo "HASH MISMATCH"; exit 1; }

echo "== 4. profiler $PROFILER_SHA"
[ -x .venv/bin/adtc-profiler ] || {
  rm -rf .venv
  python3 -m venv .venv
  .venv/bin/python -m pip install -q --upgrade pip wheel
  .venv/bin/python -m pip install "git+https://github.com/Africa-Deep-Tech-Foundation/adtc-profiler.git@$PROFILER_SHA"
}
source .venv/bin/activate

echo "== 5. measure"
PIN=""; [ "$(nproc)" -gt 4 ] && PIN="taskset -c 0-3"; echo "pin: ${PIN:-none}"
$PIN adtc-profiler run --submission . --mode participant --skip-accuracy --output smoke.json
$PIN adtc-profiler run --submission . --mode participant --output submission.json

echo "== 6. export"
{ head -2 /etc/os-release; uname -r; nproc; free -h; lscpu | grep -E "Model name|^CPU\(s\)" || true; git log -1 --format=%H; } > machine.txt
tar czf "$HOME/agrillm-profile-results.tgz" submission.json smoke.json machine.txt sha.txt -C "$HOME" agrillm-profile.log
if command -v cmd.exe > /dev/null 2>&1; then
  WINHOME="$(wslpath "$(cmd.exe /c 'echo %USERPROFILE%' 2>/dev/null | tr -d '\r')")" || true
  [ -n "${WINHOME:-}" ] && cp "$HOME/agrillm-profile-results.tgz" "$WINHOME/Desktop/" && echo "copied to Windows Desktop"
fi
cat submission.json
echo "== DONE: ~/agrillm-profile-results.tgz"
