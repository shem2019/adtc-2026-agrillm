## Inspiration

My parents recently retired from teaching and started farming.

They have land and they have time, but they don't have the things agricultural
advice usually assumes. There's no reliable internet at the farm. There's no
budget for a monthly AI subscription on a teacher's pension. There's no machine
capable of running a large model, and no extension officer arriving to answer
the question that matters this week — what is eating the maize, and what can be
done about it before the weekend.

When I read the ADTC brief, it described their situation almost exactly: the
8 GB laptop already sitting on the desk, no cloud, no discrete GPU, no API fees.
I entered because the competition and the problem I actually wanted to solve
turned out to be the same thing.

## What it does

AgriLLM is a 940 MB language model that answers crop, pest, livestock and soil
questions **completely offline**. It runs on a second-hand laptop with the
network switched off, and it's fine-tuned on East African agronomy — the pests,
seasons and crops that matter in Kenya, not generic advice written for somewhere
else.

The clearest example of what the fine-tuning bought: shown a field description
of ragged holes, windowpane scarring and moist sawdust-like frass in the maize
whorl, the base model says "maize weevil." AgriLLM says **fall armyworm
(*Spodoptera frugiperda*)** — the pest that has dominated Kenyan maize since
2017, and the correct answer.

It also learned when *not* to answer. Ask it for a pesticide dose and it tells
you to read the product label and confirm with your local extension office,
because registrations differ by country and a wrong rate can poison someone.

## How I built it

**Model selection by measurement, not reputation.** I benchmarked seven
candidates from 0.5B to 4B, on both my Mac and an x86 VM matching the audit
environment. Qwen3-4B was the most capable model I tested and I rejected it: at
5.97 tok/s it forfeits two-thirds of the throughput score to buy accuracy within
noise of a 1.5B. Qwen2.5-1.5B won because it was top-two under every throughput
assumption I could construct.

**A corpus assembled from three openly-licensed datasets** (Apache-2.0, CC0,
MIT) plus 10,564 East African entries I generated and validated — 18,248 unique
examples in total. One promising dataset was excluded because it declared no
licence.

**LoRA fine-tuning with MLX**, then fusing, converting to GGUF and quantising to
Q4_K_M. Final measurements on a 4 vCPU x86 VM: **10.4 tok/s, 1.65 GB peak RSS of
a 7 GB budget**, reproduced across two independent runs.

## Challenges I ran into

**The loss curve lied to me.** My first fine-tune had a beautiful training
curve — loss fell from 2.15 to 0.51 in 100 iterations — and produced the worst
model I built. It answered one clause of a four-part question in 19 words, and
started reciting my training data's refusal template as if it were advice.
Fitting that fast means memorising format, not learning content. I built a
comparison harness that runs my actual submitted prompts through base and
fine-tuned models side by side, and made every decision after that on
measurement rather than loss.

**The best-scoring model was the wrong one.** An IQ4_XS quantisation scored 4.4
points higher than what I shipped — 31% faster, 40% less memory. Asked to
identify fall armyworm, it answered "the Letticea leaf miner," a species that
doesn't exist. 4.25-bit quantisation had destroyed the domain knowledge while
improving every telemetry metric. I shipped the slower model.

**Reading data beats reading logs.** Five separate bugs were invisible in summary
statistics. The worst: a minimum-length filter deleted 11,095 rows of
quantitative agronomy because answers like `5.87 kg/ha` are only ten characters
— while *keeping* that dataset's "cannot be determined" refusals, which are
longer. It inverted my corpus so that 27% of training examples were refusals. I
only found it by printing two actual training examples and reading them.

**Swahili defeated me, and I'm reporting it honestly.** There's a +15% bonus for
African-language support and I tried hard to earn it. I generated 2,500 pairs,
measured that they contained only 121 distinct answers recycled 20 times each,
and threw them away. I regenerated 880 verified pairs and trained on them. The
model produced degenerate repetition — the same noun phrase repeating until the
token budget ran out. I confirmed it at two training durations. 880 rows at 4%
of a corpus teaches a 1.5B model vocabulary without grammar. **I declined the
bonus rather than claim something that would fail when a judge tested it.**

## What I learned

That most of this work is measurement, not modelling. Every time I trusted an
assumption — about the scoring formula, about my hardware, about my own data — it
was wrong in a way that only showed up when I measured it. And that knowing when
to reject your own better-scoring result is part of the engineering, not separate
from it.

## What's next

Fact-checking the generated corpus against KALRO and FAO material — the model
still confabulates crop variety names, and that's inherited from training data I
didn't verify. Swahili done properly, with enough data to teach grammar rather
than vocabulary. And offline retrieval over a farmer's own records, so the advice
knows what was planted in that field last season.

My parents' farm is where this gets tested next.

---

**Full technical report, benchmarks and reproducibility instructions:**
[REPORT.md](https://github.com/shem2019/adtc-2026-agrillm/blob/main/REPORT.md)
