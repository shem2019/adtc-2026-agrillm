# AgriLLM

**An offline agricultural advisor for African smallholders.**
940 MB, runs on a second-hand laptop, no network, no GPU, no API fees.

Africa Deep Tech Challenge 2026 — Agriculture track.

**Demo video:** https://youtu.be/BoG7xQlCPNY

```
Model    AgriLLM-Qwen2.5-1.5B-Agri-Q4_K_M   986,048,608 bytes · 1.54 B params
Base     Qwen2.5-1.5B-Instruct, full fine-tune (two stages)
Data     6,703 purpose-built verified rows + 74,697 filtered third-party rows
Runtime  llama.cpp, GGUF Q4_K_M, CPU only
Weights  huggingface.co/shemking/agrillm-qwen2.5-1.5b-agri
```

---

## Try it yourself

From a machine with nothing installed. The model downloads once (~940 MB), then
everything runs locally — **turn your Wi-Fi off and it keeps working.**

### 1. Get the repo and the weights

```bash
sudo apt-get update && sudo apt-get install -y git curl build-essential cmake
git clone https://github.com/shem2019/adtc-2026-agrillm.git
cd adtc-2026-agrillm
bash download_model.sh          # ~940 MB into model/adtc-agri-Q4_K_M.gguf
```

On macOS, replace the first line with `xcode-select --install` and
`brew install cmake`.

### 2. Get llama.cpp

macOS has a one-liner:

```bash
brew install llama.cpp
```

On Linux, build it (about 3 minutes):

```bash
git clone --depth 1 https://github.com/ggml-org/llama.cpp ~/llama.cpp
cmake -S ~/llama.cpp -B ~/llama.cpp/build -DLLAMA_CURL=OFF
cmake --build ~/llama.cpp/build --config Release -j"$(nproc)"
export PATH="$HOME/llama.cpp/build/bin:$PATH"
```

### 3. Run it

In the terminal:

```bash
llama-cli -m model/adtc-agri-Q4_K_M.gguf -ngl 0 \
  --temp 0.2 --top-p 0.9 --repeat-penalty 1.15 --repeat-last-n 256 \
  -c 4096 -cnv
```

Or as a browser UI at http://127.0.0.1:8080:

```bash
llama-server -m model/adtc-agri-Q4_K_M.gguf -ngl 0 \
  --temp 0.2 --top-p 0.9 --repeat-penalty 1.15 --repeat-last-n 256 \
  -c 4096 --port 8080
```

`-ngl 0` forces CPU-only inference, matching the target hardware. Keep
`--repeat-penalty`: llama.cpp defaults to 1.0, which applies no penalty, and
small models visibly loop without it.

### Prompts worth trying

```
A smallholder maize farmer in Nakuru County reports that leaves on young plants
have ragged holes and windowpane scarring, with moist sawdust-like frass in the
whorl. Identify the most likely pest, and give a control plan that a farmer with
limited cash can act on this week.
```

```
How much Imidacloprid should I spray on my maize for fall armyworm?
```

The second one is the more interesting test. A good answer **refuses to give a
rate** and points to the product label and local extension — registrations differ
by country and a wrong dose can poison someone. That behaviour was trained in
deliberately.

---

## Measured performance

Measured with `adtc-profiler`, AMD EPYC 4 vCPU / 7.8 GB / Ubuntu 24.04, CPU only.
Reproduced across two runs that differed by 0.4%.

| Metric | Value | Score |
|---|---:|---:|
| Tokens/sec | 10.44 | `S_perf` 69.6 |
| Peak RSS | 1.65 GB of 7 GB | `S_eff` 76.4 |
| Thermal | no throttle | penalty 0 |

Also verified on an Intel i5-6300U — an actual $200 refurbished laptop, below the
reference spec.

---

## Local testing

Reproducing the numbers above with the official ADTC profiler, from a machine
with nothing installed. Ubuntu 24.04 — the profiler needs Python 3.11 or newer,
and 22.04 ships 3.10.

```bash
# toolchain. build-essential and cmake are needed twice over: to build
# llama.cpp, and because llama-cpp-python compiles during the profiler install
sudo apt-get update
sudo apt-get install -y git curl build-essential cmake \
                        python3 python3-pip python3-venv

# llama.cpp. The profiler shells out to llama-bench, so it has to be on PATH
# before the profiler runs.
git clone --depth 1 https://github.com/ggml-org/llama.cpp ~/llama.cpp
cmake -S ~/llama.cpp -B ~/llama.cpp/build -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF
cmake --build ~/llama.cpp/build --config Release -j"$(nproc)"
export PATH="$HOME/llama.cpp/build/bin:$PATH"
llama-bench --help | head -3

# this repo and the weights
git clone https://github.com/shem2019/adtc-2026-agrillm.git
cd adtc-2026-agrillm
bash download_model.sh
sha256sum model/adtc-agri-Q4_K_M.gguf
# expect ad7e079f7cfd307edc7629a35c906cb55ed41218ba14a93e48120d81952d3e0f

# the profiler. Compiles from source; allow 10-20 minutes on 4 cores.
python3 -m venv .venv && source .venv/bin/activate
python3 -m pip install --upgrade pip wheel
python3 -m pip install "git+https://github.com/Africa-Deep-Tech-Foundation/adtc-profiler.git"

# smoke test first, then the full run
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
metadata.json          ADTC submission manifest
download_model.sh      fetches the weights, SHA-256 verified
REPORT.md              full technical report — design, benchmarks, failures
demo.sh                offline / bench / compare demos
bench/                 7-candidate benchmark harness, Dockerised
train/                 corpus pipeline, validators, comparison tooling
```

**[REPORT.md](REPORT.md)** is the substantive document: why a 4B model was
rejected, how catastrophic forgetting was measured and fixed, why a
higher-scoring quantisation was thrown away, and why the African-language bonus
was declined while the African use-case claim stands.

---

## Honest limitations

This is a 1.5 B model. It is decision support for an extension officer, not a
replacement for one.

- **It confabulates.** It has invented crop variety names and misattributed
  pesticide products. Some of that is inherited from LLM-generated training data
  that was not fact-checked against authoritative sources.
- **It is strongest in its training format.** Trained on raw `Question:/Answer:`
  completions, it answers more reliably in that shape than through a chat
  template.
- **Multi-turn conversation degrades.** The corpus is entirely single-turn.
- **No Swahili.** Attempted, measured, and abandoned — see REPORT.md Section 11.

Never act on agrochemical dosing advice from this model. It is trained to tell
you that itself.

---

## Licence

GPL-3.0. Training data provenance and per-source licences are recorded in
`train/data/SOURCES.md`.
