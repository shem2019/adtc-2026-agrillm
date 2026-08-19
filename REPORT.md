# Offline Agricultural Advisory LLM for Commodity Laptops

**Team:** TODO · **Domain:** Agriculture · **Track:** ADTC 2026 Laptop LLM

> Skeleton. Judges and the LLM audit system read this file — keep every claim
> factual, specific, and backed by a number or a citation. One to three pages.
> Delete these quote blocks before submitting.

---

## 1. Problem and context

> Who is the user, where are they, and what breaks today? Be concrete about the
> African context — this is where the African Use Case Bonus (up to 10 points)
> is won or lost. Name the country, the crop, the extension-officer-to-farmer
> ratio, the connectivity reality. Cite sources.

Target user: agricultural extension officers and smallholder farmers in East
Africa, working on farms without reliable connectivity or grid power.

- TODO: extension coverage ratio, with citation
- TODO: rural connectivity and data-cost figures, with citation
- TODO: what an officer does today when they don't know an answer

## 2. Constraints

| Constraint | Reality | Consequence for the design |
|---|---|---|
| Compute | 4 cores, integrated graphics, no discrete GPU | CPU-only inference, `-ngl 0` |
| Memory | 8 GB total, 7 GB budget | Model file ≤ ~1.5 GB after quantization |
| Connectivity | Intermittent to absent in the field | Zero network calls at inference |
| Power | Battery, often no reliable mains | Sustained load must not thermally throttle |
| Data | Little digitised African agronomy in open corpora | Corpus construction was a large share of the work |

## 3. Design decisions

> The judges want to see alternatives *considered and rejected*, with reasons.
> This section is where a thoughtful submission separates itself.

### Base model selection

TODO: paste the `score.py` ranking table. Explain that `S_perf = min(TPS/15, 1)`
is capped, so the objective was the largest model still clearing 15 tok/s rather
than the fastest model available — and show the numbers that led to the choice.

| Candidate | Params | Peak RAM | TPS | S_acc | S_total |
|---|---|---|---|---|---|
| TODO | | | | | |

### Quantization

TODO: which formats were compared and what each cost in accuracy vs. size.

### Alternatives rejected

- TODO: e.g. larger model at lower precision — rejected because…
- TODO: e.g. sub-1B model — rejected because the throughput cap made the speed
  advantage worthless while accuracy fell N points.

## 4. Tools

| Tool | Why |
|---|---|
| llama.cpp / GGUF | Required by the challenge; mmap keeps RSS near file size |
| TODO fine-tuning stack | |
| adtc-profiler | Reference measurement, ensures audit reconciliation |

## 5. Data

> Every source, its licence, and how it was cleaned. Volume in tokens.

## 6. Benchmarks

> Report the environment precisely — CPU model, RAM, OS, thread count — and
> state that measurements were taken CPU-only under a 4 vCPU / 7.5 GB cap.

**Measurement environment:** TODO

| Metric | Value | Score contribution |
|---|---|---|
| Tokens/sec (generation) | TODO | S_perf = TODO |
| Peak RSS | TODO | S_eff = TODO |
| Time to first token | TODO | — |
| Max core temperature | TODO | P_thermal = TODO |
| arc_easy (or hidden subset proxy) | TODO | S_acc = TODO |

Before/after fine-tuning comparison:

| | Base | Fine-tuned |
|---|---|---|
| Domain accuracy | TODO | TODO |
| General reasoning (arc_easy) | TODO | TODO |

> Include the general-reasoning row even if it went down. Judges trust a report
> that shows a tradeoff more than one that reports only wins.

## 7. Build in action

> Screenshots or a short clip: the model answering a real extension question on
> a laptop with networking disabled. Showing the network off is the whole point.

## 8. Limitations and honest failure modes

> Do not skip this. Name what the model gets wrong, where advice could be unsafe
> (agrochemical dosing, livestock treatment), and what you would do with more
> time. Overclaiming is the fastest way to lose a judge.

## 9. Reproducibility

```bash
git clone <repo> && cd <repo>
bash download_model.sh
adtc-profiler run --submission . --mode participant --output submission.json
```

## References

1. TODO
