# AgriLLM

**An offline agricultural advisor for East African smallholders.**
940 MB, runs on a second-hand laptop, no network, no GPU, no API fees.

Africa Deep Tech Challenge 2026 — Agriculture track.

```
Model    AgriLLM-Qwen2.5-1.5B-Agri-Q4_K_M   986,048,512 bytes · 1.54 B params
Base     Qwen2.5-1.5B-Instruct, LoRA fine-tuned on 18,248 agronomy examples
Runtime  llama.cpp, GGUF Q4_K_M, CPU only
Weights  huggingface.co/shemking/agrillm-qwen2.5-1.5b-agri
```

---

## Try it yourself

Two commands. The model downloads once (~940 MB), then everything runs locally —
**turn your Wi-Fi off and it keeps working.**

```bash
git clone https://github.com/shem2019/adtc-2026-agrillm
cd adtc-2026-agrillm
bash download_model.sh
```

Then either a browser UI:

```bash
llama-server -m model/adtc-agri.gguf -ngl 0 \
  --temp 0.2 --top-p 0.9 --repeat-penalty 1.15 --repeat-last-n 256 \
  -c 4096 --port 8080
```

…and open http://127.0.0.1:8080 — or straight from the terminal:

```bash
llama-cli -m model/adtc-agri.gguf -ngl 0 \
  --temp 0.2 --top-p 0.9 --repeat-penalty 1.15 --repeat-last-n 256 -c 4096
```

`-ngl 0` forces CPU-only inference, matching the target hardware. Don't skip
`--repeat-penalty`: llama.cpp defaults to 1.0, which is no penalty at all, and
small models visibly loop without it.

You'll need llama.cpp — `brew install llama.cpp` on macOS, or build from
[ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp).

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
higher-scoring quantisation was thrown away, and why an African-language bonus
was declined.

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
- **No Swahili.** Attempted, measured, and abandoned — see REPORT.md §8.

Never act on agrochemical dosing advice from this model. It is trained to tell
you that itself.

---

## Licence

GPL-3.0. Training data provenance and per-source licences are recorded in
`train/data/SOURCES.md`.
