# Fine-tuning plan — agriculture domain

Budget: **5 GPU-hours** (Udutech credits). That is enough for LoRA on a
1.5–2B base, and not enough for anything else. Plan accordingly.

---

## Train for how you are actually scored

`accuracy.py` scores the model two ways, and they reward different things:

- `loglikelihood` — multiple-choice tasks. The model sees `context +
  continuation` as **raw text with no chat template**, and the answer is
  whichever continuation has the highest log-probability. Instruction-following
  and politeness contribute nothing.
- `generate_until` — generative tasks, greedy decoding at `temperature=0.0`,
  scored on `exact_match`-style metrics. Terse, correctly-formatted answers win.

Neither path applies a system prompt or chat wrapper. So:

**Do not train a chatty assistant.** Train a model that (a) knows agronomy facts
densely and (b) puts high probability on correct short continuations. A model
that opens every answer with "Certainly! Here's a helpful overview…" is spending
probability mass on tokens that earn zero points and slow generation down.

Mix roughly:
- **60% domain knowledge in completion form** — declarative agronomy prose.
  Raises the log-probability of correct facts, which is what MCQ ranking reads.
- **25% MCQ-shaped items** — question, options, single-letter or short answer,
  formatted exactly as ARC/MMLU render them.
- **15% short-form instruction data** — enough to keep the model coherent for the
  judges' qualitative read of your 2 submitted + 2 hidden prompts, without
  making it verbose.

Hold out ~500 items you never train on. Evaluate with the real profiler, not
with training loss.

---

## Corpus sources

Prefer openly-licensed, verifiable text. Cite every source in `REPORT.md` — the
LLM audit system reads it.

| Source | What it gives | Note |
|---|---|---|
| FAO knowledge repositories / ECHOcommunity | Crop, livestock, agronomy manuals | Check licence per document |
| CGIAR / CIMMYT / IITA / ILRI publications | Africa-specific varieties, pests, practices | Strongest African-relevance signal |
| National extension handbooks (KALRO Kenya, NAADS Uganda, TARI Tanzania) | Exactly the register your users need | Often PDF — needs extraction |
| Plantwise / CABI pest factsheets | Pest and disease identification | Maps directly to test prompt 1 |
| [Masakhane](https://github.com/masakhane-io) | African-language parallel and NLP data | For the `sw` language claim |

**Do not scrape indiscriminately.** A small, clean, verified corpus beats a large
noisy one at this parameter count, and the report has to survive judge scrutiny.

## On the Swahili claim

`metadata.json` currently declares `language_scope: ["en", "sw"]` and
`african_alpha_claim: true`. Only keep `sw` if you actually train on Swahili
agricultural text and can show benchmark evidence. An unsupported claim is worse
than a narrower honest one — the African Use Case Bonus is judged, and judges
read the report against the artifact. Decide by day 4 and make the metadata
match reality.

---

## Recipe

Base: sweep winner (expected ~1.5–1.7B). LoRA, not full fine-tune.

```
rank            16-32          # 32 if the corpus is large and clean
alpha           2x rank
target modules  q,k,v,o,gate,up,down   (all linear; attention-only underfits at this size)
lr              1e-4 to 2e-4, cosine schedule, ~3% warmup
epochs          2-3            # small corpora overfit fast at 1.5B
seq len         2048           # matches the profiler's _N_CTX
precision       bf16
```

Use `unsloth` or `peft` + `trl`. Merge the adapter into the base before GGUF
conversion — llama.cpp cannot consume a bare LoRA in the submission path.

**Guard against catastrophic forgetting.** The hidden validation subset may not
be purely agricultural, and general reasoning ability is what carries the MCQ
score. Benchmark `arc_easy` before and after fine-tuning; if it drops more than
a couple of points, lower the learning rate or the epoch count. A model that
gained 5 points on agronomy and lost 8 on general reasoning is a net loss.

---

## Order of operations

1. Sweep first — pick the base from measured numbers, not from reputation.
2. Build and clean the corpus **before** touching the GPU. The 5 hours are for
   training, not for discovering your data is malformed.
3. One short pilot run (~30 min) to validate the pipeline end to end, including
   GGUF conversion and a profiler pass.
4. Then the real run.
5. Quantization sweep on the result.

Keep at least one GPU hour in reserve. You will want a second attempt.
