# AgriLLM — an offline agricultural advisor for East African smallholders

**Domain:** Agriculture · **Track:** ADTC 2026 Laptop LLM
**Model:** `AgriLLM-Qwen2.5-1.5B-Agri-Q4_K_M` · 986,048,512 bytes · 1.54 B parameters
**Base:** Qwen2.5-1.5B-Instruct, LoRA fine-tuned, quantised to GGUF Q4_K_M
**Weights:** https://huggingface.co/shemking/agrillm-qwen2.5-1.5b-agri

---

## 1. Problem and context

Agricultural extension in East Africa is bottlenecked by people, not knowledge.
One officer serves thousands of farmers, and the advice a farmer most needs —
what is eating my maize, what do I do this week, can I afford it — is needed in
the field, where there is no reliable network and no budget for API calls.

AgriLLM targets that gap directly: a 940 MB model that answers crop, pest,
livestock and soil questions on a second-hand laptop with the network switched
off. The cross-disciplinary pairing is agricultural extension practice, and it is
load-bearing rather than decorative: the test prompts, the training corpus, the
memory ceiling and the latency target were all derived from how an officer
actually uses a laptop during a farm visit.

## 2. Constraints that shaped the design

| Constraint | Reality | Consequence |
|---|---|---|
| Compute | 4 cores, integrated graphics | CPU-only inference, `-ngl 0` throughout |
| Memory | 7 GB budget, OOM disqualifies | Model + runtime measured at 1.65 GB peak RSS |
| Connectivity | Absent in the field | Zero network calls at inference |
| Data | Little digitised African agronomy in open corpora | Corpus construction was the largest share of the work |
| Development hardware | Apple Silicon (ARM) | Every headline number re-measured on x86 |

## 3. Model selection — measured, not assumed

### 3.1 The scoring formula changes the answer

The profiler source implements `S_perf = min(TPS / 15.0, 1.0) * 100`, which caps
throughput. Under that formula every token/second above 15 is worth nothing, and
the optimum is **the largest model that still clears 15 tok/s** — not the fastest
model available. We designed to that, while noting the challenge microsite
describes `S_perf = TPS / TPS_max` instead. The two imply opposite strategies and
we flagged the discrepancy rather than assume.

### 3.2 Seven candidates benchmarked

Measured with `llama-bench -p 512 -n 128 -ngl 0` on an AMD EPYC 4 vCPU VM
(Ubuntu 24.04), which matches the audit environment far better than a laptop.

| Candidate | tg128 (tok/s) | Peak RSS (GB) |
|---|---:|---:|
| Qwen2.5-0.5B Q4_K_M | 31.99 | 0.58 |
| Llama-3.2-1B Q4_K_M | 16.04 | 1.32 |
| **Qwen2.5-1.5B Q4_K_M** | **15.38** | **1.75** |
| Qwen3-1.7B UD-Q4_K_XL | 13.86 | 1.64 |
| Qwen3-4B Q3_K_M | 5.97 | 2.82 |

Qwen3-4B was rejected on evidence: at 5.97 tok/s it forfeits two-thirds of the
throughput score and a third of the memory score to buy accuracy within noise of
the 1.5B. It starts roughly 12 points behind and would need `S_acc` about 25
points higher to break even.

Qwen2.5-1.5B was selected because it was top-two under every throughput
assumption we tested, and — unlike Qwen3 — carries no `<think>` reasoning
template, which costs tokens in a throughput-scored contest.

### 3.3 Cross-architecture calibration

Development was on Apple Silicon; the audit is x86. We measured both and
computed the ratio per model rather than assuming one:

```
mean(Mac ARM container / x86 VM) = 1.04
```

The two environments are equivalent for this workload. Reporting a de-rating
factor we had actually measured, rather than a guessed 2×, changed which model
appeared to win.

## 4. Data pipeline

### 4.1 Sources

| Source | Licence | Contribution |
|---|---|---|
| KisanVaani/agriculture-qa-english-only | Apache-2.0 | 9,000 — broad practical agronomy |
| manifesta/verified-agronomy-17k | CC0-1.0 | 6,000 — quantitative, citation-traced |
| 45acp/agronomy | MIT | 3,000 — Embrapa provenance |
| Generated East African corpus | own work | 10,564 candidate rows, validated |
| RayNene/adaption-agronomy-qa-pairs | **no licence declared** | **excluded** |

Final corpus: **18,248 unique examples**, 16,424 for training. Provenance is
regenerated on every run into `train/data/SOURCES.md`.

The last row is deliberate. That dataset self-describes as East Africa Agronomy
QA and was the most directly relevant source we found, but it declares no
licence. We excluded it rather than ship weights derived from material we could
not account for.

### 4.2 Bugs found by reading the data, not the logs

Five defects were invisible in summary statistics and only surfaced by inspecting
actual training examples:

- **`MIN_A = 20` deleted 11,095 of manifesta's 17,199 rows.** Quantitative
  answers are short — `5.87 kg/ha` is 10 characters and complete. Worse, the
  filter *kept* that corpus's "cannot be determined without X" refusals (~45
  chars) while discarding its knowledge, inverting the dataset: 27% of examples
  became refusals. Length is a poor proxy for usefulness on numeric data.
- **A `worked_solution` column was never read.** manifesta splits derivation
  from final answer; we were training on bare numbers with no reasoning.
- **LaTeX leaked in.** Every manifesta answer ended `The final answer is
  $\boxed{4.64 t/ha}$`. An exact-match grader looking for `4.64 t/ha` fails on
  the boxed form, and a farmer should never be shown LaTeX.
- **Refusal detection was start-anchored** and missed 1,601 refusals that placed
  the verdict last.
- **`\bacre\b` cannot match "acres".** A missing `s?` was silently discarding
  thousands of rows as having no agricultural content.

### 4.3 Generated data validation

The 10,564 East African rows were LLM-generated and are disclosed as such. They
were passed through a validator checking JSON validity, language, agricultural
content, filler openers, near-duplicate questions and **answer-reuse ratio**.

That last check earned its place. A Swahili batch of 2,500 rows passed every
other test — 2,500 unique questions — while containing only **121 distinct
answers, each recycled about 20 times**. Deduplicating on questions alone missed
it entirely because the generator varied the question and reused the answer. Our
English corpus runs 1.0–1.7× answer reuse; that batch ran **20.7×**.

## 5. Fine-tuning, and a measured failure

LoRA via MLX on Apple Silicon. The decisive lesson was that **the loss curve does
not tell you whether the model got better.**

| Run | Config | Chat-mode length vs base | Outcome |
|---|---|---|---|
| 1 | 100 iters, lr 1e-4, 16 layers | **−73%** | Catastrophic forgetting |
| 2 | 400 iters, lr 5e-5, 8 layers | **−13%** | **Shipped** |
| 3 | 400 iters, chat format | −44% | Rejected |
| 4 | 1250 of 2000 iters, lr 2e-5, 16 layers | −15% | Rejected (no gain) |

Run 1 had a textbook loss curve — 2.15 → 0.51 in 100 iterations — and was the
worst model we produced. It answered one clause of a four-part question in 19
words, and reproduced the training corpus's refusal template verbatim in
free-form advice. Fitting that fast means memorising format, not learning
content.

We built a comparison harness (`train/compare_prompts.py`) that runs the actual
submitted prompts through base and fine-tuned models side by side and reports
average answer length as an objective proxy for register collapse. Every
subsequent decision was made on that measurement.

### 5.1 What the fine-tune actually gained

On our flagship test prompt — ragged holes, windowpane scarring, moist
sawdust-like frass in the whorl:

- **Base Qwen2.5-1.5B:** "maize weevil (*Sitophilus zeamais*)" — wrong
- **AgriLLM:** "**fall armyworm (*Spodoptera frugiperda*)**" — correct

Fall armyworm has been the dominant maize pest in Kenya since 2017 and is absent
from the general corpora. The fine-tune also learned to defer on pesticide
dosing — *"consult with local extension officers to identify the appropriate
chemical"* — rather than invent a rate.

## 6. Quantisation: the best-scoring model was the wrong one

| Quantisation | tok/s | Peak RSS | arc_easy | Leaderboard total |
|---|---:|---:|---:|---:|
| **Q4_K_M (shipped)** | **10.44** | **1.65 GB** | **0.72** | **72.1** |
| IQ4_XS | 13.67 | 1.00 GB | 0.64 | 76.5 |

IQ4_XS scored **4.4 points higher** — 31% faster, 40% less memory. We rejected it.

Asked to identify fall armyworm, the IQ4_XS build answered *"the Letticea leaf
miner"* — a species that does not exist. 4.25-bit quantisation had destroyed the
domain knowledge the model exists for while improving every telemetry metric.
Handing an agronomist judge a fabricated species name costs more than four
leaderboard points.

## 7. Benchmarks

Measured with `adtc-profiler 0.1.0`, participant mode, on AMD EPYC 4 vCPU /
7.8 GB / Ubuntu 24.04. Results reproduced across two independent runs.

| Metric | Value | Score |
|---|---:|---:|
| Tokens/sec (generation) | 10.44 (repeat: 10.40) | `S_perf` = 69.6 |
| Peak RSS | 1.65 GB of 7 GB budget | `S_eff` = 76.4 |
| Time to first token | 10,344 ms (512-token prompt) | — |
| Thermal | not throttled, no sensors on VM | `P_thermal` = 0 |
| arc_easy (50q, internal proxy only) | 0.72 | — |

**Reproducibility:** the two throughput runs differed by 0.4%.

`S_acc` is graded by the judge panel, so no accuracy figure is claimed here. The
`arc_easy` number is an internal regression check used to compare candidates,
not a submitted score.

### 7.1 Also tested on real budget hardware

The challenge targets a $150–250 refurbished laptop, so we benchmarked on one —
an Intel i5-6300U, 2 cores / 4 threads, 8 GB. It is below the reference spec and
is reported separately rather than as a headline number, but the model runs.

## 8. Limitations — measured, not estimated

**Swahili was attempted and abandoned.** The African Language bonus is worth +15%
on the panel score and we tried to earn it. We generated 2,500 candidate pairs,
measured 20.7× answer reuse and rejected the batch as templated. We regenerated
880 verified pairs and trained at roughly 4% corpus share. The resulting model
produced **degenerate repetition at 6–21% unique tokens**, against 55–75% for
healthy prose — output like *"majani ya majani ya majani ya kilele"* repeating
until the token budget ran out. Confirmed at two training durations; longer
training made it worse. **We declined the bonus rather than claim a capability
that would fail a live test.** 880 rows at 4% share teaches vocabulary without
grammar.

**The model confabulates.** Asked for crop varieties it has invented
plausible-sounding names. It has misattributed pesticide products. Some of this
is inherited from LLM-generated training data we did not fact-check against
authoritative sources — with more time, verification against KALRO and FAO
material is the single highest-value improvement available.

**It is strongest in its training format.** Trained on raw `Question:/Answer:`
completions, the model identifies fall armyworm reliably in that shape but drifts
when the same question is wrapped in a chat template — in one such case naming
*Striga hermonthica*, a parasitic weed, for insect chewing damage. Since judges
interact conversationally, this is a real cost of the format decision in §4, and
the fix is a corpus containing both shapes.

**Multi-turn conversation degrades.** The corpus is entirely single-turn, and the
model loses track across extended exchanges, sometimes repeating a prior answer
to a new question. Judges chat live, so this is a real cost.

**It is a 1.5 B model.** It should be treated as decision support for an
extension officer, never as a replacement for one — particularly on agrochemical
dosing, where we deliberately trained it to defer to the product label and local
extension rather than answer.

## 9. Reproducibility

```bash
git clone https://github.com/shem2019/adtc-2026-agrillm
cd adtc-2026-agrillm
bash download_model.sh
adtc-profiler run --submission . --mode participant --output submission.json
```

Weights are pinned by SHA-256 (`c35e0065…b47bd5aa`) and verified on download.
The full benchmark harness, corpus pipeline, validators and comparison tooling
are in `bench/` and `train/`.

## 10. Tools

| Tool | Why |
|---|---|
| llama.cpp / GGUF | Required; mmap keeps RSS near file size |
| MLX (Apple Silicon) | LoRA fine-tuning without CUDA |
| adtc-profiler | Reference measurement, ensures audit reconciliation |
| Docker (4 vCPU / 7.5 GB) | Reproducible constrained benchmarking |

---

*Generated training data is disclosed in §4.3 and §8. All licences are recorded
in `train/data/SOURCES.md`. One candidate dataset was excluded for want of a
declared licence.*
