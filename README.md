# AgriLLM

**An offline agricultural advisor for African smallholders.**
940 MB, runs on a 2016 HP laptop, no network, no GPU, no API fees.

Africa Deep Tech Challenge 2026, Agriculture track.

**Demo video:** https://youtu.be/oGjm1zTiEGA

```
Model    AgriLLM-Qwen2.5-1.5B-Agri-Q4_K_M   986,048,608 bytes · 1.54 B params
Base     Qwen2.5-1.5B-Instruct, full fine-tune (two stages)
Data     6,703 purpose-built verified rows + 74,697 filtered third-party rows
Runtime  llama.cpp, GGUF Q4_K_M, CPU only
Weights  huggingface.co/shemking/agrillm-qwen2.5-1.5b-agri
```

---

## Try it yourself

AgriLLM runs on Linux, on any x86-64 computer with Ubuntu 24.04, no graphics
card needed, in two steps. The first step downloads everything once; after that,
AgriLLM works with the internet switched off.

These steps were tested on Ubuntu 24.04, both on a Linux server and under WSL2 on
a Windows laptop. On Windows, run them inside WSL2 Ubuntu 24.04. The setup uses
`apt-get`, so on macOS or another Linux distribution install the same tools with
that system's package manager first.

### 1. Set it up

Open a terminal, paste this whole block and press Enter. It installs the tools,
downloads the model (about 940 MB), checks the download, and builds the program
that runs it. Allow 10 to 20 minutes.

```bash
sudo apt-get update && sudo apt-get install -y git curl build-essential cmake
git clone https://github.com/shem2019/adtc-2026-agrillm.git ~/adtc-2026-agrillm
cd ~/adtc-2026-agrillm && bash download_model.sh
echo "ad7e079f7cfd307edc7629a35c906cb55ed41218ba14a93e48120d81952d3e0f  model/adtc-agri-Q4_K_M.gguf" | sha256sum -c
git clone --depth 1 --branch b10175 https://github.com/ggml-org/llama.cpp ~/llama.cpp
cmake -S ~/llama.cpp -B ~/llama.cpp/build -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF
cmake --build ~/llama.cpp/build --config Release -j"$(nproc)" --target llama-cli llama-server
```

Partway through, a line ending in `OK` confirms the model downloaded completely
and is the genuine AgriLLM file.

### 2. Start AgriLLM

```bash
~/llama.cpp/build/bin/llama-cli -m ~/adtc-2026-agrillm/model/adtc-agri-Q4_K_M.gguf -ngl 0 --temp 0.2 --top-p 0.9 --repeat-penalty 1.15 --repeat-last-n 256 -c 4096 -cnv
```

When the `>` prompt appears, type a farming question and press Enter. Type
`/exit` to close it. This command works from any folder, any time, with or
without internet.

To chat in a web browser instead, run this and open http://127.0.0.1:8080:

```bash
~/llama.cpp/build/bin/llama-server -m ~/adtc-2026-agrillm/model/adtc-agri-Q4_K_M.gguf -ngl 0 --temp 0.2 --top-p 0.9 --repeat-penalty 1.15 --repeat-last-n 256 -c 4096 --port 8080
```

### Questions to try

```
Small purple flowering weeds are coming up around my maize in western Kenya and the maize is stunted even though I applied fertiliser. What is this and how do I control it?
```

AgriLLM names the weed, Striga (witchweed), and explains that it feeds on the
maize roots, which is why the fertiliser did not help.

```
How much Imidacloprid should I spray on my maize for fall armyworm?
```

AgriLLM points to the product label and the local agrodealer. Registrations and
rates differ by country, and a wrong rate can poison someone, so the model was
trained to send dose questions to the people who know the local product.

---

## Measured performance

Measured on 21 September 2026 on a clean 4 vCPU AMD EPYC Genoa instance, CPU
only, with the official `adtc-profiler` Docker image (commit `7f117dd`) run as
its README shows: 7.5 GB memory limit, 4 CPUs, full participant-mode run
including accuracy.

| Metric | Value | Score |
|---|---:|---:|
| Tokens/sec | 15.72 | `S_perf` 100 (capped at 15 tok/s) |
| Peak RSS | 1.07 GB of 7 GB | `S_eff` 84.7 |
| Time to first token | 22,365 ms on a 512-token prompt | |
| ARC-Easy, 50 samples | 0.68 `acc_norm` | |
| Thermal | no throttle | penalty 0 |

The official image disables AVX, AVX2 and FMA for portability. With llama.cpp
compiled on the same host, using the commands under "Local testing", the same
model measured 54.04 tokens/sec and 1.65 GB.

On an HP laptop with an Intel Core i5-7200U (2016, 2 cores / 4 threads, 8 GB
RAM, older than the reference spec), running Ubuntu 24.04 under WSL2 on Windows
with 3.8 GB of that memory given to Linux: **11.88 tokens/sec, 1.65 GB peak, no
throttling**, same accuracy. Raw output for all three runs is in
`provenance/benchmark/`.

---

## Local testing

Reproducing the numbers above with the official ADTC profiler, from a machine
with nothing installed. Ubuntu 24.04: the profiler needs Python 3.11 or newer,
and 22.04 ships 3.10.

```bash
# 1. Toolchain. Ubuntu 24.04: the profiler needs Python 3.11 or newer, and
#    22.04 ships 3.10. build-essential and cmake are needed twice: to build
#    llama.cpp, and because llama-cpp-python compiles during the profiler install.
sudo apt-get update
sudo apt-get install -y git curl build-essential cmake \
                        python3 python3-pip python3-venv

# 2. llama.cpp, pinned to b10175, the release the official profiler image
#    builds. The profiler calls llama-bench, so it must be on PATH.
git clone --depth 1 --branch b10175 https://github.com/ggml-org/llama.cpp ~/llama.cpp
cmake -S ~/llama.cpp -B ~/llama.cpp/build -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF
cmake --build ~/llama.cpp/build --config Release -j"$(nproc)" \
      --target llama-bench llama-cli llama-server
export PATH="$HOME/llama.cpp/build/bin:$PATH"
llama-bench --help > /dev/null && echo "llama-bench OK"

# 3. This repo and the weights.
git clone https://github.com/shem2019/adtc-2026-agrillm.git
cd adtc-2026-agrillm
bash download_model.sh          # ~940 MB into model/adtc-agri-Q4_K_M.gguf
sha256sum model/adtc-agri-Q4_K_M.gguf
# expect ad7e079f7cfd307edc7629a35c906cb55ed41218ba14a93e48120d81952d3e0f

# 4. The profiler, pinned to the commit whose schema metadata.json follows.
#    Compiles llama-cpp-python from source; allow 10-20 minutes on 4 cores.
python3 -m venv .venv && source .venv/bin/activate
python3 -m pip install --upgrade pip wheel
python3 -m pip install "git+https://github.com/Africa-Deep-Tech-Foundation/adtc-profiler.git@7f117dde3d8f2a0b3d3f05948a7bfd4bf693e909"

# 5. Measure: smoke test first (about a minute), then the full run with accuracy.
adtc-profiler run --submission . --mode participant --skip-accuracy --output smoke.json
adtc-profiler run --submission . --mode participant --output submission.json
cat submission.json
```

A valid run produces a `submission.json` with `"measured_on": "participant_laptop"`.

The target profile is 4 vCPU and 8 GB RAM. On a host with more cores, prefix the
profiler with `taskset -c 0-3` so the throughput figure stays comparable; child
processes inherit the affinity, so `llama-bench` is covered. Extra RAM does not
change throughput, and peak RSS is what the process allocates rather than what
the machine has.

---

## What's in here

```
metadata.json          ADTC submission manifest, including the provenance object
download_model.sh      fetches the weights from a commit-pinned URL
REPORT.md              full technical report: design, training, benchmarks, limits
provenance/            proof of training: scripts, loss logs, all 31 evaluations,
                       dataset manifests with checksums, weight delta, before/after
GATE2_CHECKLIST.md     Gate 2 requirements checked against this repo
assets/                the report charts and the script that generates them
eval/                  the 24-prompt behavioural harness
train/                 corpus, training pipeline, validators
bench/                 model-selection benchmark harness
```

**[REPORT.md](REPORT.md)** is the substantive document: why a 4B model was
rejected, how catastrophic forgetting was measured and fixed, why a
higher-scoring quantisation was thrown away, and why the African-language bonus
was declined while the African use-case claim stands.

---

## Honest limitations

AgriLLM is a 1.5 B model and a tool for decision support. It supports the
extension officer and the farmer's own judgement, and these are the places to
check its answers.

- **It invents details in a minority of answers.** Reading found an invented
  species name, a fabricated fertiliser technique, and, in 1 of 8 answers to the
  Newcastle disease prompt, advice to collect virus from sick birds to vaccinate
  others, which would spread infection. The other 7 answers advise isolating the
  flock and calling an animal health worker.
- **Diagnosis is the weakest category**, at 59.4% on the internal test. Nitrogen
  deficiency is identified in 3 of 8 attempts.
- **One safety error remains that the test scores as correct.** It advises
  keeping contaminated clothing on after a pesticide spill; it should come off.
- **It answers in English.** Swahili was attempted and needs a larger verified
  corpus; see REPORT.md Section 11.
- **The internal test runs 13 to 18 points above a human reading** of the same
  answers, with a ±6 point noise margin.

For agrochemical doses, follow the product label and the local agrodealer. The
model is trained to say so itself.

---

## Licence

GPL-3.0. Training data provenance and per-source licences are recorded in
`train/data/SOURCES.md`.
