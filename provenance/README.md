# provenance/ — proof of training

Everything Gate 2 Section 3.1 asks for. Section 3.1 lists adapter weights for
LoRA and QLoRA teams; AgriLLM is a full fine-tune, so the equivalent evidence
here is the training scripts and configs that produced the model, per-step loss
logs for every run, the dataset manifests with checksums, a tensor-by-tensor
weight-delta report against the base model, and the published full-precision
checkpoints from which the shipped GGUF can be re-derived byte for byte.

## What's here

| Path | What it is |
|---|---|
| `before_after.md` | **Gate 2 Section 3.1 before/after comparison** — three prompts, base model vs shipped model, full category table |
| `checksums.txt` | SHA256 of the base model, every published model file, the full-precision checkpoints, the source dataset, and each of the 25 corpus files |
| `weight_delta.json` | Tensor-by-tensor comparison against the base model: **99.9986% of parameters changed** |
| `training/` | Every script and config that produced the model, unmodified |
| `logs/` | Per-step loss for all 7 runs, full trainer records, and all 31 evaluation results |
| `dataset/` | Corpus manifest with per-file checksums, a 50-row representative sample, and the stage-1 filter manifest |

## Reproducing the model

On a bare Ubuntu GPU box with at least 40 GB VRAM and an NVIDIA driver already
present (`nvidia-smi` should work):

```bash
sudo apt-get update && sudo apt-get install -y git jq
git clone https://github.com/shem2019/adtc-2026-agrillm.git
cd adtc-2026-agrillm
bash train/pipeline/reproduce.sh --all
```

`reproduce.sh --stage setup` installs everything else the pipeline needs —
CUDA toolkit, the C++ toolchain, a Python virtualenv, PyTorch, transformers and
llama.cpp built with CUDA.

`setup_gpu.sh` installs the CUDA toolkit if missing, verifies the C++ toolchain,
keeps the CUDA build of `torch` in place against llama.cpp's requirements file,
and writes the base model commit to `base_model_pin.json`, which the training
scripts read. `preflight.sh` then asserts twelve conditions in about thirty
seconds.

About five hours end to end on a single A6000.

## The runs, in order

| Run | Init from | Data | Epochs | LR | Final train loss | Best checkpoint |
|---|---|---|---|---:|---:|---|
| `lr5e-6` | base | ours (6,703) | 4 | 5e-6 | 1.473 | 612 — 59.3% |
| `lr1e-5` | base | ours | 4 | 1e-5 | 1.155 | final — 60.3% |
| `lr2e-5` | base | ours | 4 | 2e-5 | 0.886 | 408 — 63.9% |
| `lr5e-6-2ep` | base | ours | 2 | 5e-6 | 1.816 | 408 — 51.7% |
| `stage1` | base | AI71ai filtered (74,697) | 1 | 1e-5 | 1.908 | — |
| **`2stage`** | stage1 | ours | 6 | 2e-5 | **0.594** | **1224 — 78.9% ← SHIPPED** |
| `2stage-x` | 2stage/1224 | ours | 4 | 1e-5 | 0.021 | final — 81.7%, **not shipped** |

The last row scored highest and was rejected as memorisation; its training loss
of 0.021 is the reason. Full detail in REPORT.md Section 8.

## Files the evaluator may want first

- `logs/results_summary.csv` — all 31 evaluations, one row each, with the
  per-category breakdown
- `logs/loss_stage2.csv` — per-step loss for the run that produced the shipped
  model
- `logs/run_record_stage2.json` — trainer state: wall clock, FLOPs, GPU, every
  hyperparameter as actually applied
- `dataset/stage1_manifest.json` — what was kept and cut from the third-party
  corpus, with counts

## Attribution

Stage 1 used **[`AI71ai/agrillm-train-146k`](https://huggingface.co/datasets/AI71ai/agrillm-train-146k)**,
a third-party dataset published on the Hugging Face Hub by AI71ai under
Apache-2.0, and used here under that licence. It was filtered before use, and
exactly what was removed is recorded in `dataset/stage1_manifest.json` and
REPORT.md Section 5. The name overlap is coincidental — the dataset was found
after this project was named.

Base model: **[`Qwen/Qwen2.5-1.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)**
@ `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`, Apache-2.0, by Alibaba Cloud.

Quantisation and inference: **[llama.cpp](https://github.com/ggerganov/llama.cpp)**, MIT.
Training: **[transformers](https://github.com/huggingface/transformers)**, Apache-2.0.

The corpus (`train/african/_clean/`, 6,703 rows) is original work, released
CC-BY-4.0.
