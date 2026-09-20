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

**3. llama.cpp's convert requirements silently replaced CUDA torch with a CPU
build.** This one was a pipeline bug, not the image. `setup_gpu.sh` installed
`torch` (CUDA) early, then later ran
`pip install -r requirements-convert_hf_to_gguf.txt`. That file pins torch *and*
points at PyTorch's CPU wheel index, because GGUF conversion only needs to read
tensors. The result was `torch 2.11.0+cpu` replacing `2.14.0+cu130`, with no
error at install time. It surfaced at the start of the first training run as
`CUDA not available`, which looked like a driver failure and was not:
`nvidia-smi`, `/proc/driver/nvidia/version` and `libcuda.so.1` were all healthy
throughout. Fixed by reinstalling torch, and in the pipeline by filtering torch
and index directives out of that requirements file plus a hard post-install
guard that fails setup rather than letting the problem surface an hour later.

Net: roughly 45 minutes of billed time lost — two thirds to a base image that
had a GPU driver but no compiler toolchain, one third to our own dependency
ordering.

### The NaN divergence, and how it was actually found

Every training attempt died the same way: loss around 2.55 with a finite
gradient norm at step 10, then a jump to ~530-550 and `grad_norm=nan` by step
20, identically at 5e-6, 1e-5 and 2e-5. Three separate fixes were attempted by
changing one variable and re-running, and all three failed — two of them because
the box was still on stale code and the fix had never executed at all.

That approach was abandoned for `train/pipeline/diagnose.py`, which runs 220
real micro-batches of real data through forward and backward under four
numerical configurations and reports the first batch where anything goes
non-finite. Measured result:

| params | attention | compute | outcome |
|---|---|---|---|
| fp32 | sdpa | bf16 autocast | grad nan at micro-batch 18 |
| fp32 | **eager** | bf16 autocast | survived 220, loss 2.49 -> 2.21 |
| fp32 | sdpa | **pure fp32** | survived 220, loss 2.46 -> 2.22 |
| bf16 | sdpa | bf16 | grad nan at micro-batch 18 |

**Cause: SDPA attention combined with bf16 compute produces non-finite
gradients on right-padded batches.** It needs both; changing either fixes it.
The original configuration (bf16 params + sdpa) and the first attempted fix
(fp32 + autocast + sdpa) reached the same failure by different routes, which is
why the fix appeared to change nothing.

Two things this also settled, both of which had been guessed at wrongly:
transformers 4.57.6 honours `dtype=`, `torch_dtype=` and no kwarg at all — all
three load fp32, so the deprecation warning was noise and the dtype was never
the problem. And the corpus is clean: token ids in range, zero rows with no
trainable tokens.

Shipped configuration is `attn_implementation: eager`. It should not be changed
back to sdpa or flash_attention_2 without re-running `diagnose.py`.

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
