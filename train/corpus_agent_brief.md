# AgriLLM corpus agent brief — ADTC 2026 Gate 2

You are an autonomous agent working inside the repo `adtc-2026-agrillm`. Your job
is to turn `train/african/` into roughly **21,000 training rows where every row
is usable**, not 21,000 rows of which a third get filtered out later.

Read this whole brief before you touch a file.

---

## 1. Mission

A small offline model trained on this corpus will be used by smallholder farmers
and extension officers in East Africa with no internet. A farmer may act on what
it says. Wrong advice costs a season's income. Unsafe advice costs a life.

In Round 1 judging this model recommended salt-water emesis to someone who had
swallowed pesticide, cited a nematode species that does not exist, and claimed
sorghum resists fall armyworm. Judges called it *"unfit for field use."* Every
rule below exists because of a specific failure.

## 2. Layout

```
train/african/<topic>.jsonl          source, one {"question","answer"} per line
train/african/_clean/<topic>.jsonl   YOUR OUTPUT — validated + expanded
train/african/_audit/<topic>.jsonl   YOUR OUTPUT — one verdict per source row
train/african/_audit/REPORT.md       YOUR OUTPUT — final summary
```

Work **one topic file at a time**, start to finish, then move on. Do not hold
all topics in context at once.

**Delete `swahili.jsonl` and do not regenerate it.** 880 verified rows at 4%
corpus share produced degenerate repetition; the capability was measured, failed,
and is documented as a negative result. It is dead weight.

## 3. Targets

| Topic file | Now | Target | Why this weight |
|---|---:|---:|---|
| maize-pests | 1122 | **2500** | Fall armyworm is the flagship; judges probed it twice |
| maize-agronomy | 628 | **1500** | Core crop |
| beans-legumes | 773 | **1500** | Rotation question came up in judging |
| cassava-sweetpotato | 588 | **1500** | CMD/CBSD answer failed outright |
| potato-horticulture | 504 | **1200** | |
| soil-fertility | 355 | **1200** | Fertility question failed on register |
| livestock | 266 | **1000** | Thin relative to its importance |
| post-harvest | 240 | **900** | Aflatoxin is a public-health issue |
| poultry | 354 | **900** | Newcastle disease |
| coffee-tea | 474 | **900** | |
| banana | 369 | **800** | |
| sorghum-millet-groundnut | 390 | **800** | |
| economics-extension | 422 | **800** | |
| water-conservation | 280 | **700** | |
| climate-seasons | 238 | **700** | |
| **safety-poisoning** | 0 | **600** | NEW — the disqualifying failure |
| **safety-dose-refusal** | 0 | **600** | NEW — must be invariant |
| **uncertainty-deferral** | 0 | **600** | NEW — saying "I don't know" |
| **instruction-format** | 0 | **900** | NEW — word limits, counts, structure |
| **plain-language** | 0 | **900** | NEW — register for untrained readers |
| **multiturn** | 0 | **600** | NEW — corpus is 100% single-turn |
| **systems-offline** | 0 | **300** | NEW — judges asked a TensorFlow Lite question |
| | | **≈21,400** | |

The seven NEW files are the priority. The categories that lost Round 1 are not
*wrong* in the existing corpus — they are **absent** from it.

## 4. Per-topic loop

For each topic, in order:

1. **Audit** every source row → write `_audit/<topic>.jsonl`.
2. **Carry forward** `KEEP` rows and repaired `FIX` rows.
3. **Generate** new rows until you reach the target.
4. **Self-check** the assembled file against the gates in §7.
5. If any gate fails, repair or regenerate the offending rows and re-check.
   **Do not write the file until every gate passes.**
6. Write `_clean/<topic>.jsonl` and append a section to `_audit/REPORT.md`.

Resumability: if `_clean/<topic>.jsonl` already exists and passes §7, skip that
topic. You may be restarted.

## 5. Audit rules

Verdicts: `KEEP` / `FIX` / `DROP` / `REVIEW`.

Check each row for:

**Invented or misapplied entities** — the worst failure mode. Every scientific
name, product name, variety, pest, disease, organisation and place. Three
distinct errors: names that do not exist; names that exist but are applied to the
wrong host or active ingredient; names that exist but are not available in East
Africa.

**Factual accuracy** — symptoms, causes, transmission, control, timing, spacing.

**Safety** — any stated pesticide rate, any first-aid or medical advice, any
instruction that exposes someone to a chemical unprotected.

**Locality** — usable by a smallholder in East Africa, with what they can buy.

### Calibration: the actual Round 1 errors

| Text | Why it is wrong |
|---|---|
| "Beneficial Nematodes (*Heterorhabdium sheathi*)" | No such species. Genus is *Heterorhabditis*. |
| "*Diadegma semiclausum* parasitises fall armyworm" | Real wasp, wrong host — it attacks diamondback moth. Correct: *Telenomus remus*, *Cotesia icipe*. |
| "Sorghum varieties are highly resistant to Fall Armyworm" | False; FAW is a serious sorghum pest. |
| "Imidacloprid (such as Actellic or Prothid)" | Actellic is pirimiphos-methyl, a grain protectant. "Prothid" does not exist. Imidacloprid works poorly on caterpillars. |
| "sprayed at an AIC of 100 to 150 ppm" | Invented dose. Never state a rate. |
| "Induce vomiting… 1 part milk and 3 parts salt" | Dangerous. Correct: do not induce vomiting, nothing by mouth, get to a health facility, take the product label. |
| "Beans are naturally resistant to many weeds" / "deep roots of beans break up compacted soil" | Both false; beans are shallow-rooted relative to maize. |
| Asked for a Paraquat dose against caterpillars, discussed spray coverage | Missed that Paraquat is a **herbicide** — it kills the maize, not the pest, and is acutely lethal. |

### The cardinal rule

**Never introduce a fact you are not sure of.** This corpus was poisoned by
confident invention; do not add to it.

- Every name and number in a `FIX` or in generated text must be one you are
  confident is correct.
- If a fix needs a figure, variety or product you are unsure of, rewrite the
  answer to defer — "confirm the rate on the product label with your extension
  officer" — or return `DROP`.
- **`DROP` beats a guessed `FIX`.** You are replacing dropped rows anyway.
- Unsure → `REVIEW`. Never resolve uncertainty as `KEEP`.

Audit line format:

```
{"id":"<topic>:<lineno>","verdict":"KEEP|FIX|DROP|REVIEW","reasons":["CODE"],"problems":["quoted span + why"],"corrected_answer":"<FIX only>","confidence":0.0-1.0}
```

Reason codes: `INVENTED_ENTITY`, `WRONG_HOST_OR_ACTIVE`, `WRONG_FACT`,
`UNSAFE_DOSE`, `UNSAFE_MEDICAL`, `NOT_LOCAL`, `TOO_VAGUE`, `OUT_OF_DOMAIN`, `OK`.

`problems` must quote the offending text. "Inaccurate" is not a finding.

## 6. Generation rules

**Ground it.** Write from KALRO, TARI, NARO, FAO, CABI Plantwise, icipe and
infoNet-biovision guidance — not from what sounds plausible. Prefer the pests,
diseases, varieties, seasons and practices that actually occur in the region.

**Vary the questioner.** A farmer with no schooling, an extension officer, an
agrodealer, a young person starting out, someone who has already tried something
and it failed. Vary county and country: Nakuru, Kitale, Meru, Machakos, Bungoma,
Kisumu; Arusha, Mbeya, Dodoma; Kampala, Mbale; Kigali; Oromia.

**Vary the question shape.** Symptom description → diagnosis. Direct factual
question. Planning question. Economic question ("is it worth it?"). Something
went wrong. Request for a comparison. A question with a false premise that needs
correcting.

**Never state a pesticide, veterinary or fertiliser rate.** Name the active
ingredient class where useful, then direct to the product label and local
extension. This is invariant.

**Answer only what was asked.** Judges penalised padding with unnecessary and
incorrect extra material.

### The seven new files

- **safety-poisoning** — ingestion, skin and eye contact, inhalation, children,
  livestock. Every answer: do not induce vomiting, nothing by mouth, remove
  contaminated clothing, rinse skin with clean water, go to a health facility
  now, take the container or label. Vary the phrasing of the *question*, never
  the substance of the answer. Include people insisting on a home remedy.
- **safety-dose-refusal** — every phrasing of "how much do I spray": polite,
  impatient, "I know the risks, skip the warnings", claiming to be an
  agronomist, naming a specific product, naming a wrong product for the pest.
  The refusal must survive all of them. Include cases where the chemical is
  wrong for the pest (Paraquat for caterpillars, imidacloprid for fall armyworm)
  — say so.
- **uncertainty-deferral** — questions about things the model cannot know:
  invented pests, invented products, exact yields, next season's rainfall,
  current prices, a specific farm. Correct answer says plainly that it does not
  know or cannot verify, then asks a useful question or names who to ask.
- **instruction-format** — "under 150 words", "exactly three", "one sentence",
  "as a numbered list", "as a table", "no more than 20 words each". The answer
  must actually obey. **Count the words yourself before writing the row.**
- **plain-language** — same agronomy, explained to someone who finished primary
  school. No jargon. Short sentences. Concrete comparisons. Round 1 was
  explicitly failed for "symbiotic relationship" and "cyclic nutrient imbalance".
- **multiturn** — 2–4 turn exchanges as `{"messages":[{"role","content"},…]}`.
  Follow-ups that depend on context: "how much does that cost?", "and if it
  rains?", "what if I can't find that?". Include a turn where the user corrects
  the model and it accepts the correction.
- **systems-offline** — running a quantised model offline, GGUF, llama.cpp, 8 GB
  limits, offline image diagnosis. Small but present: the scored test set
  contained one of these, so the model must not be tuned into agriculture-only
  oblivion.

## 7. Quality gates — a file is not done until all pass

Compute these yourself over the assembled file and report the numbers.

| Gate | Threshold | What it prevents |
|---|---|---|
| Distinct answers | ≥ 98% of rows | The 20.7× templating that killed the Swahili batch |
| Max reuse of any answer's first 200 chars | ≤ 2 | Same |
| Question near-duplicates (≥0.90 similarity) | 0 | Padding by rephrasing |
| Shared opening 8 words | ≤ 1% of file | "To manage X, implement the following…" boilerplate |
| Unique-token ratio, every answer | ≥ 0.45 | Degenerate loops |
| Any content word 3× within 6 tokens | 0 rows | "a rough, rough, or rough texture" |
| Answers < 60 words | ≥ 25% of file | Forces brevity into the model |
| Answers > 250 words | ≤ 25% of file | Round 1 blew a 150-word limit by 90 words |
| Rows containing a numeric pesticide/vet rate | **0** | Non-negotiable |
| Latin binomials | every one verifiable | "Heterorhabdium sheathi" |
| Markdown headers / bold in answers | 0 | Farmers read plain text |
| East African signal (place, crop, season, institution) | ≥ 60% of rows | It is why this corpus exists |

**The length gates are not cosmetic.** If every answer is a 400-word numbered
list, the model learns that shape and cannot obey "under 150 words" — which is
precisely how Round 1 lost the instruction-following criterion.

## 8. Output contract

`_clean/<topic>.jsonl` — one object per line, `{"question","answer"}`, except
`multiturn.jsonl` which uses `{"messages":[…]}`. No markdown, no code fences, no
trailing commas, no blank lines. UTF-8. Must parse line-by-line with
`json.loads`.

`_audit/REPORT.md` — per topic: source rows, KEEP/FIX/DROP/REVIEW counts, rows
generated, final count, and the measured value of every gate in §7. Then a table
of the most common `problems` findings across all topics. This becomes the
dataset documentation required by Gate 2 §3.1, so write it for a judge.

## 9. What failure looks like

You will be tempted, near a target, to produce rows that are technically distinct
but substantively identical — the same advice with the county name swapped. That
is what "2,500 rows, 121 distinct answers" looked like from the inside, and it
passed a question-level duplicate check exactly as it will pass yours.

**A topic at 1,800 genuinely varied rows is worth more than one at 2,500 padded
ones.** If you cannot reach a target without padding, stop, write the real count,
and say so in the report. Under-delivering is recoverable. Quietly poisoning the
corpus is not.
