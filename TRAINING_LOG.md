# Training log — Gate 2 run

A dated record of the run that produced the submitted model: environment,
decisions, failures and fixes. Kept because Gate 2 §3.1 asks for training logs
and because a reviewer reproducing this needs to know what the machine actually
was, not what it was supposed to be.

Machine-readable artefacts live in `provenance/` (assembled by
`train/pipeline/capture_provenance.py`). This file is the narrative.

---

## 2026-09-20 — Environment

Rented a single H100 on Shadeform (AGH Cloud partner GPU), des moines region,
$3.822/hr.

```
Host      shadecloud, Ubuntu 22.04.5 LTS, kernel 6.8.0-90-generic
CPU       Intel Xeon Platinum 8352Y @ 2.20GHz, 20 cores
RAM       125 GB
Disk      1.3 TB ext4 (14 G used at start)
GPU       NVIDIA H100 PCIe, 81559 MiB, driver 580.126.09
torch     2.14.0+cu130, CUDA 13.0, device reported as 79.2 GB, sm_90
CUDA tk   12.6.85 (nvcc) — installed manually, see below
llama.cpp ggml 0.24.0, commit 3cf0325, built with GGML_CUDA=ON
```

bf16 matmul sanity check on the card passed before any training was attempted.

### Base model, pinned

```
repo    Qwen/Qwen2.5-1.5B-Instruct
commit  989aa7980e4cf806f80c7fef2b1adb7bc71aa306
license apache-2.0
params  1,543,714,304 (BF16)
```

The commit SHA was resolved by `setup_gpu.sh` at download time and independently
verified against the HuggingFace model API before being written into
`metadata.json` and `train/pipeline/sft_config.yaml`. Pinning matters: `main`
moves, and an unpinned base makes the run unreproducible.

### Two setup failures worth recording

Both were environment gaps on the rented image, not pipeline bugs, but they cost
billed time and the next person should not pay for them twice.

**1. CUDA Toolkit absent.** `nvidia-smi` reported CUDA 13.0 and PyTorch worked
immediately, which is misleading: the driver was present and torch ships its own
bundled runtime, but there was no `nvcc`, so llama.cpp's CUDA build failed with
`CUDA Toolkit not found`. Fixed by adding NVIDIA's apt repo and installing
`cuda-toolkit-12-6`. Toolkit 12.6 against a 580 driver is fine — drivers are
backward compatible — and it does not conflict with torch's cu130 wheels, since
the toolkit is only used to compile llama.cpp.

**2. `g++` absent despite `build-essential`.** After nvcc was installed the build
still failed, with `gcc: fatal error: cannot execute 'cc1plus'`. The C++
compiler backend was missing, so nvcc could not preprocess host code. Fixed with
`apt-get install --reinstall build-essential g++ gcc`. CMake then reported
`CUDA host compiler is GNU 11.4.0` and the build completed.

Net: roughly 30 minutes of billed time lost to a base image that had a GPU
driver but no compiler toolchain.

---

## Corpus as trained

25 files, 6,703 rows, all passing `train/verify_corpus.py`.

This is a much smaller number than the 22,300 rows an earlier generation process
claimed. That corpus was ~700 underlying answers wrapped in `Ref N:` serial
prefixes: 100% distinct as raw strings, ~3% by shape. It was discarded. The
recovered genuine corpus then had real defects of its own — stated pesticide and
veterinary doses, banned PPE abstractions, and place-swap padding — which were
fixed or deduplicated rather than shipped. Every row here is one that survived a
mechanical gate and, for the safety categories, was written by hand.

---

## Run record

<!-- Append as stages complete. Keep the numbers, not just the narrative. -->

| Stage | Started | Outcome |
|---|---|---|
| setup | 2026-09-20 09:30 | complete after the two fixes above |
| gates | | |
| data | | |
| train | | |
| select | | |
| quant | | |
| verify | | |
| provenance | | |
