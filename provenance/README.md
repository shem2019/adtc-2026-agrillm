# provenance/ — proof of training

Everything Gate 2 §3.1 asks for. AgriLLM is a **full-parameter fine-tune**, so
there are no LoRA/QLoRA adapters to submit; §3.1 lists adapter weights only for
teams who used LoRA. In their place this folder carries the training scripts and
configs that produced the model, per-step loss logs for every run, the dataset
manifests with checksums, a tensor-by-tensor weight-delta report against the base
model, and the published fp32 checkpoints from which the shipped GGUF can be
re-derived byte for byte.

## What's here

| Path | What it is |
|---|---|
| `before_after.md` | **§3.1 before/after comparison** — three prompts, base model vs shipped model, full category table |
| `checksums.txt` | SHA256 of the base model, every published GGUF, the fp32 checkpoints, the third-party dataset, and each of our 25 corpus files |
| `weight_delta.json` | Tensor-by-tensor comparison against the base model: **99.9986% of parameters changed** |
| `training/` | Every script and config that produced the model, unmodified |
| `logs/` | Per-step loss for all 7 runs, full trainer records, and all 31 evaluation results |
| `dataset/` | Corpus manifest with per-file checksums, a 50-row representative sample, and the stage-1 filter manifest |

## Reproducing the model

One command on a bare GPU box with at least 40 GB VRAM:

```bash
git clone https://github.com/shem2019/adtc-2026-agrillm.git
cd adtc-2026-agrillm
bash train/pipeline/reproduce.sh --all
```

No file needs editing, no variable needs exporting. `setup_gpu.sh` installs the
CUDA toolkit if absent, verifies the C++ toolchain actually works rather than
trusting the package manager, guards `torch` against being silently downgraded
to a CPU build by llama.cpp's own requirements file, and writes the pinned base
model commit to `base_model_pin.json` — which the training scripts read, so the
pin never has to be hand-edited into a tracked file. `preflight.sh` then asserts
twelve conditions in about thirty seconds; each one exists because its absence
cost real GPU time.

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

The last row is the one worth reading the note about: it scored highest and was
rejected. Full reasoning in REPORT.md §9.

## Files the evaluator may want first

- `logs/results_summary.csv` — all 31 evaluations, one row each, with the
  per-category breakdown
- `logs/loss_stage2.csv` — per-step loss for the run that produced the shipped
  model
- `logs/run_record_stage2.json` — trainer state: wall clock, FLOPs, GPU, every
  hyperparameter as actually applied
- `dataset/stage1_manifest.json` — what we kept and cut from the third-party
  corpus, with counts

## Attribution

Stage 1 used **[`AI71ai/agrillm-train-146k`](https://huggingface.co/datasets/AI71ai/agrillm-train-146k)**,
a third-party dataset published on the Hugging Face Hub by AI71ai under
Apache-2.0. It is not our work. We filtered it before use and record exactly what
was removed in `dataset/stage1_manifest.json` and REPORT.md §6. The name overlap
with our project is coincidental — we found the dataset after naming AgriLLM.

Base model: **[`Qwen/Qwen2.5-1.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)**
@ `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`, Apache-2.0, by Alibaba Cloud.

Quantisation and inference: **[llama.cpp](https://github.com/ggerganov/llama.cpp)**, MIT.
Training: **[transformers](https://github.com/huggingface/transformers)**, Apache-2.0.

Our own corpus (`train/african/_clean/`, 6,703 rows) is original work, released
CC-BY-4.0.
