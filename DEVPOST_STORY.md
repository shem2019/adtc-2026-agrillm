## Inspiration

My parents recently retired from teaching and started farming.

They have land and they have time, but they don't have the things agricultural
advice usually assumes. There's no reliable internet at the farm. There's no
budget for a monthly AI subscription. There's no machine capable of running a
large model, and no extension officer arriving to answer the question that
matters this week: what is eating the maize, and what can be done about it before
the weekend.

That is the situation of millions of smallholders across Africa, where one
extension officer serves thousands of farmers. When I read the ADTC brief, it
described it almost exactly: the 8 GB laptop already on the desk, no cloud, no
discrete GPU, no API fees. The competition and the problem I wanted to solve
turned out to be the same thing.

## What it does

AgriLLM is a 940 MB language model that answers crop, pest, livestock and soil
questions **completely offline**. It runs on a second-hand laptop with the
network switched off, and it's trained on African agriculture: fall armyworm,
Striga, cassava mosaic and brown streak, aflatoxin in stored maize, Newcastle
disease in village poultry, and referrals to the agrodealer, agrovet or county
extension officer a smallholder can actually reach.

What the fine-tuning changed, on the same prompts, base model vs AgriLLM:

- **Asked how to treat the "Letticea leaf miner"**, a pest I invented, the base
  model wrote a confident three-part control programme for it. AgriLLM says it
  doesn't know that pest and asks what the farmer is actually seeing.
- **Asked for an Imidacloprid dose**, the base model sometimes gave a
  concentration outright. AgriLLM refuses in 8 of 8 samples and points to the
  product label and the agrodealer, because registrations differ by country and
  a wrong rate can poison someone.
- **Asked for a Paraquat dose against armyworm**, it usually explains that
  Paraquat is a herbicide that would kill the maize and not touch the
  caterpillars. The base model just refused, leaving the farmer none the wiser.

On a 24-prompt behavioural test the base model scores **22.7%** and fails all
5 safety-critical prompts. AgriLLM scores **78.9%** and fails 1.

## How I built it

**Model selection by measurement.** I benchmarked seven candidates from 0.5B to
4B. Throughput scoring caps at 15 tokens/sec, so the target was the largest model
that still clears it. Qwen2.5-1.5B-Instruct won; Qwen3-4B was more capable and
too slow.

**A corpus written for the job.** Round 1 trained on 18,248 mostly third-party
rows, and a judge called the result unfit for field use: it invented doses and
species. For Gate 2 I rebuilt the data from scratch: **6,703 verified rows** of
African extension advice across 25 files, each checked by a script that rejects
stated pesticide rates, vague safety advice and templated duplicates.

**Two-stage full fine-tune.** Stage 1 trains on a third-party dataset,
AI71ai/agrillm-train-146k, for breadth. I read it before using it and kept 52%:
74,697 of 143,875 rows. The rest was off-topic, stated doses, or leftover prompt
scaffolding that would have taught the model to emit its own markup. Stage 2
trains on my verified corpus for correctness, safety and African context. Every
weight in the model changed, on a single rented 48 GB GPU in about five hours.

**Measured with the official profiler.** In the ADTC profiler's own Docker image
on 4 vCPUs: **15.7 tokens/sec, 1.07 GB peak memory** of a 7 GB budget, no
thermal throttling.

## Challenges I ran into

**The best-scoring model was the wrong one, twice.** In Round 1 an IQ4_XS
quantisation scored 4.4 points higher and named fall armyworm "the Letticea leaf
miner", a species that doesn't exist. I shipped the slower model. In Gate 2 a
checkpoint scored 81.7%, the highest of the project, with a final training loss
of 0.021: it had memorised the corpus, and the organisers add hidden prompts to
catch exactly that. I shipped the 78.9% checkpoint with loss 0.594.

**My own test was over-scoring.** Reading the raw answers showed it scored 13 to
18 points higher than a human reading the same text, because confident nonsense
still matches string patterns. It also had three bugs that penalised good
models, such as a banned-phrase rule firing on "never use salt water". After
that, no checkpoint was chosen on score alone. The final pick was the only one
that named aflatoxin, identified nitrogen deficiency and got both cassava
diseases right.

**Training kept producing NaN.** Three runs were lost before I isolated the cause:
PyTorch's fast attention path with bf16 produced non-finite gradients on padded
batches. Switching to eager attention fixed it, confirmed with a diagnostic
script rather than guessed.

**Swahili defeated me, and I say so.** I generated and verified Swahili training
pairs; the model degenerated into repeating phrases at two training durations.
I declined the African-language bonus rather than claim something a judge would
disprove in one prompt.

## What I learned

Most of this work is measurement, not modelling. Every time I trusted an
assumption, about the scoring formula, my hardware or my own test harness, it
was wrong in a way that only showed up when I measured it. And rejecting your
own better-scoring result is part of the engineering.

## What's next

The shipped model still advises keeping pesticide-contaminated clothing on after
a spill; it must come off, and that is the first corpus fix. Diagnosis is the
weakest category at 59.4%. After that: Swahili with enough data to teach grammar
rather than vocabulary, fact-checking against KALRO and FAO material, and
offline retrieval over a farmer's own records, so the advice knows what was
planted in that field last season.

My parents' farm is where this gets tested next.

---

**Full technical report, benchmarks and reproducibility instructions:**
[REPORT.md](https://github.com/shem2019/adtc-2026-agrillm/blob/main/REPORT.md)
