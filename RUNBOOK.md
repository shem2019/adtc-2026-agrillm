# ADTC 2026 — Runbook

Deadline: **25 Aug 2026, 09:45 GMT+3**. Ten days from 15 Aug.

---

## What the scoring actually rewards

Read from the profiler source, not the Devpost page — they disagree.

```
S_total = 0.50*S_acc + 0.30*S_perf + 0.20*S_eff - P_thermal
S_perf  = min(TPS / 15.0, 1.0) * 100
S_eff   = max(0, (7.0 - peak_rss_gb) / 7.0) * 100
```

Three consequences that shape every decision below.

**1. Throughput is capped at 15 tok/s, not ranked against other teams.**
Devpost advertises `TPS ÷ TPS_max`, implying a race. `src/adtc_profiler/README`
and the scoring code say `min(TPS/15, 1.0)`. Above 15 tok/s the marginal token
is worth zero. Teams that chase a 135M model to win a speed race will give away
~20 points of accuracy for nothing.

**2. Accuracy is machine-graded by lm-eval log-likelihood, not judges reading prose.**
`accuracy.py` calls `lm_eval.simple_evaluate` against the quantized GGUF and
scores multiple-choice continuations by log-probability. Default task
`arc_easy`; the docstring says real audits use *"the full hidden 30% validation
subset distributed by judges."* So:
- The scored artifact is **one `.gguf` file**. No app, no RAG, no system prompt.
- The Devpost blurb's talk of retrieval over local corpora and application UX is
  never exercised by the pipeline. It only affects `REPORT.md`, the video, and
  the judges' qualitative impression.
- Optimise for MCQ ranking and calibration, not for chattiness.

**3. Peak RAM is sampled only during `llama-bench`**, which runs `-ngl 0` with a
tiny context (`-p 512 -n 128`). Peak RSS ≈ file size + overhead. Long-context KV
cache is a non-issue; don't over-engineer for it.

**Target: the largest, most accurate model that still clears 15 tok/s on a
4-vCPU x86 CPU.** That is roughly a **1.1–1.5 GB GGUF**.

---

## ⚠ The trap that will disqualify you

`adtc-profiler compare` fails a submission when participant and audit
throughput differ by **more than 50%** (flags at 25%).

Your Mac — especially Apple Silicon — has several times the memory bandwidth of
a 10th-gen i5 with DDR4-3200. Even with `-ngl 0` you may measure 45 tok/s where
their VM measures 16. That is **-64%: an automatic `fail` verdict**, for a model
that did nothing wrong.

**Therefore: produce the final `submission.json` on an x86_64 Linux VM with 4
vCPU and 8 GB RAM.** A `c6i.xlarge`-class box or any $0.05/hr equivalent for one
hour is the cheapest insurance in this competition. Use the Mac + Docker only
for *relative* ranking during the sweep.

---

## Step 1 — Sweep the candidates (tonight, ~1 hour)

Prerequisites: Docker Desktop running, `python3`, `curl`.

```bash
cd adtc-2026-agrillm/bench
./sweep.sh
```

First run builds the image (~10–15 min: llama.cpp and llama-cpp-python both
compile from source). Then each candidate downloads and profiles in ~2 minutes.

Verify the repo IDs in `candidates.tsv` on huggingface.co first — GGUF repo
layouts move, and a 404 silently wastes a slot.

Read the ranking:

```bash
python3 score.py --derate 2.0
```

`score.py` de-rates your measured TPS to estimate the audit VM, and flags any
candidate that drops under 15 tok/s after de-rating. Tune `--derate` once you
have one real x86 measurement to calibrate against.

Then, on the two or three finalists only:

```bash
./sweep.sh --accuracy      # slow: 20-40 min per model
```

This runs the same `arc_easy` subset the audit uses by default, giving you a
real `S_acc` instead of a guess.

---

## Step 2 — Fine-tune (days 2–6)

See `FINETUNE.md`. Summary: LoRA the sweep winner on an agriculture corpus
shaped like the evaluation, using the 5 free Udutech GPU hours.

---

## Step 3 — Quantize and publish

```bash
# convert merged HF model -> GGUF -> quantize
python3 llama.cpp/convert_hf_to_gguf.py ./merged --outfile adtc-agri-f16.gguf
./llama.cpp/build/bin/llama-quantize adtc-agri-f16.gguf adtc-agri.gguf Q4_K_M
```

Sweep `Q4_K_M`, `Q5_K_M`, `Q4_K_S`, `IQ4_XS` and re-rank — the quantization
choice moves all three score components at once, so it is worth a full pass.

Upload to a **public** HF repo, then update in lockstep:
- `download_model.sh` → `MODEL_URL`, `EXPECTED_SHA256`
- `metadata.json` → `model.name`, `model.quantization`, `model.parameters_estimate`

`gguf.fraud_check` compares your claimed `parameters_estimate` against the GGUF
header's real `params_count`. Report the true number.

---

## Step 4 — Final submission run (day 8–9, on x86 Linux)

```bash
bash download_model.sh
adtc-profiler run --submission . --mode participant --output submission.json
cat submission.json     # confirm "measured_on": "participant_laptop"
```

Run the **full** pass — no `--skip-accuracy`. A report with `accuracy: []`
scores zero on 50% of the leaderboard.

Sanity-check reconciliation by running the same model through the container
profile and diffing:

```bash
adtc-profiler compare submission.json audit-simulation.json
```

---

## Step 5 — Ship

- [ ] Repo **public** on GitHub
- [ ] No placeholders left in `metadata.json` (`team_id`, model fields)
- [ ] Exactly **2** `test_prompts` — organizers add 2 hidden ones
- [ ] `model/` and `*.gguf` in `.gitignore`, no weights committed
- [ ] `bash download_model.sh` succeeds from a clean clone, no credentials
- [ ] `REPORT.md` complete
- [ ] Video ≤ 2 minutes
- [ ] Submitted on Devpost before 25 Aug 09:45 GMT+3

---

## Open questions worth asking on Discord

1. Which lm-eval task family is the hidden agriculture validation subset drawn
   from? Confirms whether to train for MCQ ranking or generative `exact_match`.
2. Is the audit VM x86_64? (Assumed yes — affects the whole reconciliation plan.)
3. Is `TPS_REFERENCE = 15.0` final, or still "provisional" as Devpost says?
   If it rises, the optimal model size drops.

Answers to 1 and 3 change the build. Ask early.
