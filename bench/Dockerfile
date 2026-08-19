# Benchmark environment that mirrors the ADTC audit sandbox.
#
# The audit runs Ubuntu 22.04 on 4 vCPU / ~7.5 GB with no discrete GPU, and
# llama-bench is invoked with `-ngl 0` (CPU only). This image reproduces that
# so your submission.json reconciles with their audit.json inside the
# comparator's ±25% throughput / ±15% memory tolerance.
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# Ubuntu 22.04 ships Python 3.10; the profiler requires >= 3.11, so we install
# 3.11 alongside. python3.11-venv is what provides ensurepip.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git curl ca-certificates \
        python3.11 python3.11-dev python3.11-venv \
        libcurl4-openssl-dev \
    && rm -rf /var/lib/apt/lists/*

# --- llama.cpp -----------------------------------------------------------------
# The profiler shells out to `llama-bench`, which must be on PATH.
#
# GGML_NATIVE=OFF disables -march=native. We want a portable, deterministic
# build: native tuning would optimise for whatever host built the image, which
# makes your numbers depend on your laptop's CPU rather than on the model.
# The source arrives via COPY, already downloaded by sweep.sh on the host.
#
# Downloading it here instead would route through Docker Desktop's virtualized
# network stack, which on macOS measured ~109 KB/s against 2.4 MB/s on the host
# — a 20x penalty, and the reason the original `git clone` died with "early EOF"
# after 38 minutes. Fetching on the host and copying in removes Docker's network
# from the critical path completely.
COPY llama.cpp.tar.gz /tmp/llama.tar.gz
RUN mkdir -p /opt/llama.cpp \
    && tar -xzf /tmp/llama.tar.gz -C /opt/llama.cpp --strip-components=1 \
    && rm -f /tmp/llama.tar.gz \
    && test -f /opt/llama.cpp/CMakeLists.txt

RUN cd /opt/llama.cpp \
    && cmake -B build \
         -DGGML_NATIVE=OFF \
         -DLLAMA_CURL=OFF \
         -DBUILD_SHARED_LIBS=OFF \
         -DCMAKE_BUILD_TYPE=Release \
    && cmake --build build --config Release -j "$(nproc)" \
    && cp build/bin/llama-bench /usr/local/bin/ \
    && (cp build/bin/llama-cli /usr/local/bin/ || true) \
    && rm -rf /opt/llama.cpp

# --- ADTC profiler -------------------------------------------------------------
# Install into a venv rather than switching python3 via update-alternatives.
# The alternatives route silently breaks: apt's python3-pip is bound to 3.10, so
# after repointing python3 -> 3.11 the interpreter has no pip module at all.
# A venv gives us a self-consistent 3.11 + pip and puts both first on PATH.
RUN python3.11 -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

# llama-cpp-python compiles from source here; slow, but the layer caches.
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir \
       "git+https://github.com/Africa-Deep-Tech-Foundation/adtc-profiler.git"

# Fail the build now, not 40 minutes into a sweep, if either tool is missing.
RUN command -v llama-bench >/dev/null \
    && python3 -c "import llama_cpp, lm_eval" \
    && adtc-profiler --help >/dev/null \
    && echo "✓ llama-bench, llama-cpp-python, lm-eval and adtc-profiler all present"

WORKDIR /work
ENTRYPOINT ["/bin/bash"]
