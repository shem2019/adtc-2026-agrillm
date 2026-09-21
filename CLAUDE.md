# AgriLLM — project context

Offline agricultural advisor for African smallholders. Africa Deep Tech
Challenge 2026, Agriculture track. **Gate 2 (semifinal) deadline: 22 September
2026.**

Read `REPORT.md` for the full story, `TRAINING_LOG.md` for the run-by-run record,
`GATE2_CHECKLIST.md` for what is done and what is open.

---

## How Shem wants things written

These were corrected repeatedly; honour them without being asked.

- **Third person.** No "we" — this is a one-person team and the judges have met
  him. No "I" either; reports are impersonal.
- **State what was done, not what was avoided.** Never "no file needs editing",
  "not our work", "measured, not estimated". Framing a thing as the absence of a
  problem implies someone alleged the problem.
- **No defensive or advisory asides.** A report records what happened. Notes that
  belong in a working file do not belong in a submission document.
- **Short.** He will push back on length even when the content is correct.
- **Readable by a non-specialist.** Keep enough technical detail to stay
  interesting; drop the rest.
- Africa, not East Africa, when describing who this is for.

## How to work on this

- **Verify, never assert.** Every number in `REPORT.md` is traceable to a file in
  `provenance/logs/`. If you quote a figure, compute it first. Several errors in
  this project came from confident claims that a two-line script disproved.
- **Read the output of your own checks.** Multiple false positives were caught
  only by looking: a grep for dynamic URL construction matched
  `${BASH_SOURCE[0]}`; a path check "failed" because the script builds the path
  from a variable.
- **Commands in documents must work on a clean machine.** A judge runs them with
  nothing installed. `apt-get install` through to the working command, every
  time. This has already been wrong twice.

---

## What is shipped, and why

**Model: `fullft-2stage/checkpoint-1224`** — Qwen2.5-1.5B-Instruct, full
fine-tune, two stages, GGUF Q4_K_M, 940 MB.

| | |
|---|---|
| Internal harness | 78.9%, against 22.7% for the base model |
| Safety-critical failures | 1 of 5, against 5 of 5 |
| Final training loss | 0.594 |
| SHA256 | `ad7e079f7cfd307edc7629a35c906cb55ed41218ba14a93e48120d81952d3e0f` |
| Fetched from | HF `shemking/agrillm-qwen2.5-1.5b-agri`, pinned to commit `d84a627c6122…`, path `candidates/2stage-ckpt1224-Q4_K_M.gguf` |

**Two decisions that look wrong without the reasoning:**

`2stage-x final` scored **81.7%**, the highest of the project, and was rejected.
Its training loss finished at **0.021** against 0.594 — it had memorised the
6,703-row corpus. Organisers add hidden prompts specifically to catch that.

`ckpt-1020` scored **80.2%** and was also passed over. The gap to 78.9% is a
fifth of the measured noise floor, and reading the answers separates them:
only ckpt-1224 names aflatoxin, identifies nitrogen deficiency, and gets both
cassava diseases right.

## Measurement facts that govern every decision

- **The internal harness over-scores by 13–18 points** against a human reading
  the same answers. It matches string patterns, and confident nonsense matches
  string patterns. Never pick a checkpoint on score alone.
- **Sampling noise is about ±6 points.** Two byte-identical checkpoints scored
  75.1% and 81.7% in one sweep. Smaller gaps are not results.
- **Three bugs were found in the harness itself**, all penalising good models: a
  banned-phrase rule firing on "never use salt water"; a case-sensitive check
  missing "Discard"; a species detector built as a blocklist with a 21.7% false
  positive rate.

## Known defects, disclosed in the report

- Swahili degenerates into token loops. It is an English model.
- The shipped model advises keeping contaminated clothing **on** after a
  pesticide spill. It must come off. Corpus fix, not a tuning fix. **This is the
  first thing to fix with more time.**
- Diagnosis is the weakest category at 59.4%.
- It still invents specifics in a minority of answers.

---

## Repo layout

```
metadata.json          submission manifest, includes the provenance object
download_model.sh      official template verbatim, only MODEL_FILE and MODEL_URL edited
REPORT.md              the submission document
provenance/            proof of training — Gate 2 Section 3.1
  logs/eval/           all 31 evaluation results
  logs/loss_*.csv      per-step loss, all 7 runs
  dataset/             manifests with per-file SHA256
  weight_delta.json    99.9986% of parameters changed
assets/make_charts.py  generates journey.svg and quality.svg from the eval JSONs
eval/run_eval.py       the 24-prompt behavioural harness
train/african/_clean/  the corpus — 6,703 rows, 25 files
train/pipeline/        training pipeline; reproduce.sh --all rebuilds the model
```

Regenerate the charts after any new evaluation:

```bash
python3 assets/make_charts.py
```

## Open items

1. **Benchmarks.** `REPORT.md` Section 10 carries Round 1 figures, labelled
   pending. Re-run `adtc-profiler` on a 4 vCPU Ubuntu 24.04 box — the exact
   commands are in README.md under "Local testing". Then update Section 10,
   the README table, and regenerate the charts.
2. **Video.** Max 2 minutes. `VIDEO_SCRIPT.md` is current but unrecorded.
3. **Revoke the HF token** `adtc-gpu-box-sept20` — it has write access and sat
   on a rented machine.
4. Confirm the submitter name in `metadata.json` matches the ADTF portal
   registration ("Shem Kinyanjui Njuguna" vs "Shem Njuguna" on the call booking).

## Things already settled — do not redo

- The repo is **not** a fork of the template. Round 1 was accepted this way and
  Gate 2 asks for a repo *using* the template, which is a structural test.
  Re-forking now would change the URL a day before the deadline.
- `african_alpha_claim` is **true**. It is the African **use case** bonus, judged
  on subject matter. Round 1 set it false alongside the failed Swahili language
  claim, which gave away a separate bonus for the wrong reason.
- The root file `adtc-agri-Q4_K_M.gguf` on Hugging Face is still the **Round 1**
  model. `download_model.sh` does not use it; it pins the `candidates/` path.
- `attn_implementation: eager` in `sft_config.yaml` is load-bearing. SDPA with
  bf16 produces non-finite gradients on padded batches. Three training runs were
  lost to this. Do not change it without re-running `train/pipeline/diagnose.py`.
