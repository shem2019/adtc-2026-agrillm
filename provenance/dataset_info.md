# Dataset information

Two datasets were used, one per training stage. Both are listed in
`metadata.json` under `provenance.training_datasets`.

## Stage 1: AI71ai/agrillm-train-146k (third party)

| | |
|---|---|
| Source | https://huggingface.co/datasets/AI71ai/agrillm-train-146k |
| Publisher | AI71ai |
| Licence | Apache-2.0 |
| Snapshot | `6c10a9eef08a9ca65f87b3cfdc5e34c49da1d69e` |
| File | `train.jsonl` |
| SHA256 | `c6f7755a951eda368453ddd6e702ea5eba9eb66238be312bc24a7f30d5d43abc` |
| Rows published | 143,875 |
| Rows used | 74,697, after filtering |
| Role | broad agricultural vocabulary and coverage |

The filter is `training/prepare_stage1_external.py`. Its rules and per-rule
counts are in `dataset/stage1_manifest.json`, and REPORT_DETAILS.md Section 5 explains
each cut.

## Stage 2: AgriLLM African Extension Corpus (purpose-built)

| | |
|---|---|
| Source | https://github.com/shem2019/adtc-2026-agrillm/tree/main/train/african/_clean |
| Licence | CC-BY-4.0 |
| Files | 25 JSONL files |
| Rows | 6,703 |
| Checksums | per-file SHA256 in `dataset/corpus_manifest.json` and `checksums.txt` |
| Quality check | `training/verify_corpus.py`; every file passes |
| Role | safety behaviour, register, African context, identity |

A 50-row sample is in `dataset/corpus_sample_50.jsonl`, and the same rows are
shown with the chat template applied in `dataset/sample_rendered.txt`.

## Base model and output

| | |
|---|---|
| Base model | `Qwen/Qwen2.5-1.5B-Instruct` at commit `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`, Apache-2.0, 1,543,714,304 parameters |
| Method | full-parameter supervised fine-tune in two stages; every weight updated (`weight_delta.json`: 99.9986% of parameters changed) |
| Training hardware | 1× NVIDIA RTX A6000 48 GB for both stages; the earlier single-stage sweep ran on 1× H100 PCIe 80 GB |
| Post-training | checkpoint 1224 of stage 2, converted to GGUF F16 with llama.cpp `convert_hf_to_gguf.py`, then quantised to Q4_K_M (`training/export_gguf.sh`) |
| Shipped file SHA256 | `ad7e079f7cfd307edc7629a35c906cb55ed41218ba14a93e48120d81952d3e0f` |
| Rebuild | `bash train/pipeline/reproduce.sh --all` (GPU with at least 40 GB VRAM) |
