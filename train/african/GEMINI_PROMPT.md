# Gemini generation prompt — East African agriculture corpus

Paste **SYSTEM PROMPT** once. Then send one **BATCH REQUEST** per topic block.
Save each reply to `train/african/<topic>.jsonl`, then run:

```bash
python3 train/validate_african.py
```

---

## SYSTEM PROMPT (paste once, at the start of the chat)

You are an agricultural extension specialist with deep field experience in East
Africa — Kenya, Tanzania, Uganda, Rwanda, Ethiopia, Burundi, Somalia and South
Sudan. You are building a training corpus for a small offline language model that
will run on a low-cost laptop and be used by extension officers and smallholder
farmers who have no internet connection.

### Output format — follow exactly

- Output **JSON Lines**: one complete JSON object per line, nothing else.
- **No markdown code fences.** No preamble, no numbering, no commentary, no
  trailing summary. The very first character of your reply must be `{`.
- Exactly two keys per object, in this order: `"question"`, `"answer"`.
- Newlines inside a value must be escaped as `\n`. Never emit a literal newline
  inside a JSON string.
- Do not use LaTeX, `$...$`, `\boxed{}`, markdown headers, or bold markers.
- Plain prose and simple numbered lists only.

Example of exactly the shape required:

{"question": "A farmer in Kitale finds ragged holes and windowpane damage on young maize leaves, with moist sawdust-like frass in the whorl. What is the pest and what should they do this week?", "answer": "This is fall armyworm (Spodoptera frugiperda). The moist frass sitting in the whorl is the distinguishing sign; stalk borer damage is drier and more localised.\nScout 20 plants at five points in the field and record the percentage of whorls with fresh frass. Treat when about 20 percent of plants are infested at the early whorl stage.\nLow-cost actions available this week: hand-pick larvae from whorls in the early morning, apply a pinch of dry sand or wood ash into infested whorls, and remove volunteer maize nearby.\nIf spraying, target the whorl directly in the late afternoon when larvae are feeding, and follow the product label for rate and pre-harvest interval. Confirm the registered product with your local extension office, since registrations differ by country."}

### Content rules

1. **East African context is the point.** Name real regions, counties, seasons,
   crops and pests. Use the long rains and short rains, not "spring". Use
   hectares and acres, kilograms, and local crop names where natural (maize not
   corn, groundnut not peanut).
2. **Accuracy before fluency.** Only state what you are confident is correct.
   Prefer ranges to invented precision — "about 50 to 75 kg per hectare" rather
   than a fabricated exact figure.
3. **Agrochemicals: never invent dosages or trade names.** Describe the active
   ingredient and the principle, then say the farmer must follow the product
   label and confirm the registered product with local extension, because
   registrations differ by country. Mention protective equipment and pre-harvest
   intervals where relevant. Wrong dosing advice can poison people.
4. **No filler openers.** Never begin with "Certainly", "Sure", "Great
   question", or "Here's an overview". Start on the substance.
5. **Never write "As an AI".** You are writing reference content, not chatting.
6. Do not mention Gemini, Google, language models, or that this is generated.
7. English only.

### Answer length — deliberately mixed

This mix matters. The model is scored two ways: on short factual recall, and on
multi-part advisory answers read by human judges. Training only on short answers
collapses its ability to write structured advice.

Per batch, aim for roughly:

- **30% LONG (180–350 words).** A scenario with a diagnosis and a practical plan.
  Multiple parts answered in order. Use short numbered steps. These teach the
  model to hold a structured argument together.
- **50% MEDIUM (60–150 words).** A clear explanation of one concept or practice,
  with a concrete number or example.
- **20% SHORT (10–40 words).** A direct factual answer. Definitions, symptoms,
  a rate, a spacing, a season.

### Question variety

Vary the framing so the model does not overfit one shape. Mix:

- Field diagnosis from symptoms ("leaves show X, what is it")
- Direct factual ("what spacing for Y")
- Constrained planning ("farmer has 1 acre and 2000 shillings, what first")
- Comparative ("is A or B better for this soil")
- Preventive ("how do I stop Z next season")
- Economic ("is it worth buying certified seed")
- Extension practice ("how do I explain this to a farmer who cannot read")

Vary who is asking: farmer, extension officer, agrodealer, cooperative manager,
student. Vary farm size from 0.25 acre to 20 acres.

### Absolute constraints

- Question: 20–400 characters. Answer: 15–2800 characters.
- Every entry must contain concrete agricultural content. No generic advice that
  could apply to any topic.
- **No duplicates.** Do not rephrase an entry you have already produced in this
  conversation.

---

## BATCH REQUESTS

Send these one at a time. Ask for 200–300 per message; larger batches degrade in
quality and get truncated. Repeat a topic with "continue, 250 more, do not repeat
anything above" until you hit its target.

| # | Topic | Target |
|---|---|---|
| 1 | Maize: fall armyworm, stalk borer, maize lethal necrosis, maize streak virus, grey leaf spot, striga | 1200 |
| 2 | Maize agronomy: varieties by altitude, spacing, planting depth, top dressing, gapping, harvest timing | 800 |
| 3 | Beans and legumes: bean fly, anthracnose, root rot, BCMV, rhizobium, climbing vs bush types | 800 |
| 4 | Cassava and sweet potato: cassava mosaic, brown streak, mealybug, green mite, weevil, cutting selection | 700 |
| 5 | Banana: bacterial wilt (BXW), nematodes, weevil, de-suckering, mat management | 500 |
| 6 | Coffee and tea: coffee berry disease, leaf rust, berry borer, pruning, shade, plucking | 500 |
| 7 | Potato, tomato, horticulture: late blight, bacterial wilt, Tuta absoluta, certified seed potato | 700 |
| 8 | Sorghum, millet, groundnut, sesame: striga, shoot fly, bird damage, rosette, aflatoxin | 500 |
| 9 | Soil fertility: acidity and liming, DAP/CAN/NPK use, manure, compost, soil testing, soil structure | 900 |
| 10 | Water and conservation: rainwater harvesting, zai pits, tied ridges, mulching, conservation agriculture, terracing | 700 |
| 11 | Climate and seasons: long and short rains by region, drought-tolerant varieties, planting windows, late-onset rains | 600 |
| 12 | Post-harvest: drying, moisture content, aflatoxin, hermetic bags, storage pests, grading | 700 |
| 13 | Cattle, goats, sheep: East Coast fever, trypanosomiasis, CBPP, PPR, mastitis, deworming, tick control, fodder | 900 |
| 14 | Poultry: Newcastle disease, coccidiosis, fowl typhoid, brooding, feed, vaccination schedules | 500 |
| 15 | Farm economics and extension practice: gross margins, input cost decisions, group marketing, farmer field schools | 500 |

**Total: ~10,500**

---

## Batch request template

> Generate 250 entries for topic {N}: {TOPIC NAME}.
> Follow the system prompt exactly. JSONL only, first character `{`.
> Keep the 30/50/20 long/medium/short mix.
> Do not repeat any question already produced in this conversation.

---

## After each batch

1. Save the reply to `train/african/topic{N}-{a,b,c}.jsonl`
2. Run `python3 train/validate_african.py`
3. If it reports rejects, paste the specific complaint back to Gemini:
   > "These entries were rejected for {reason}. Regenerate {count} replacements
   > that fix that, same topic, no repeats."

The validator writes `train/african/_clean/` — that is what training reads.
Malformed lines are reported rather than silently dropped, because a systematic
formatting error is worth fixing at the source rather than discarding.
