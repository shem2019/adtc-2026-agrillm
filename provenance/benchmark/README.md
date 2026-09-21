# Benchmark run, 21 September 2026

Two runs on the same clean 4 vCPU x86-64 instance: one in the official profiler
Docker image, one with llama.cpp compiled on the host using the commands in
REPORT.md Section 12.

| File | What it is |
|---|---|
| `profiler-official-image.json` | full participant-mode run inside the official profiler Docker image (commit `7f117dd`), `--memory=7.5g --cpus=4` |
| `machine-official-image.txt` | machine and image details for that run |
| `profiler-full.json` | `adtc-profiler run --mode participant`, full run with accuracy |
| `profiler-smoke.json` | the same with `--skip-accuracy`, run first |
| `machine.txt` | OS, CPU, core count, memory, repo commit |
| `model-sha256.txt` | SHA256 of the file `download_model.sh` fetched |

Profiler commit `7f117dde3d8f2a0b3d3f05948a7bfd4bf693e909`, llama.cpp `b10175`.
The files are renamed from `submission.json` and `smoke.json` so the
template's ignore rule for local profiler output does not exclude them.
