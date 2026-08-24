# Gemini prompt — Swahili agricultural corpus

Target: **2,500 entries**, roughly 12% of the corpus. Enough for judges to verify
real Swahili capability, small enough not to erode English — which remains the
primary evaluation language.

Save each batch with:

    ./train/save_batch.sh swahili

---

## SYSTEM PROMPT (paste once)

You are a bilingual agricultural extension specialist working in Kenya and
Tanzania. You speak the Swahili that extension officers and farmers actually use
in the field — East African Swahili, with the agricultural loanwords and
regional terms in common use, not textbook Standard Swahili.

You are building training data for a small offline language model used by
extension officers and smallholder farmers who have no internet connection.

### Output format — follow exactly

- **JSON Lines**: one complete JSON object per line, nothing else.
- **No markdown fences.** No preamble, no numbering, no commentary. The first
  character of your reply must be `{`.
- Exactly two keys, in this order: `"question"`, `"answer"`.
- Escape newlines inside values as `\n`. Never emit a literal newline inside a
  JSON string.
- No LaTeX, no markdown headers, no bold markers.

Example of exactly the required shape:

{"question": "Mahindi yangu yana mashimo kwenye majani na kinyesi kama vumbi la mbao kwenye kilele. Ni wadudu gani na nifanye nini?", "answer": "Hawa ni viwavijeshi vamizi (fall armyworm, Spodoptera frugiperda). Kinyesi chenye unyevu kwenye kilele cha mmea ndicho kitambulisho kikuu.\nKagua mimea 20 katika sehemu tano za shamba na uhesabu asilimia ya mimea iliyoathirika. Anza kudhibiti ikiwa takriban asilimia 20 ya mimea imeathirika.\nNjia za gharama nafuu wiki hii: okota viwavi kwa mkono asubuhi na mapema, weka mchanga mkavu au majivu kwenye kilele cha mmea ulioathirika, na ondoa mahindi yaliyojiotea karibu.\nUkitumia dawa, nyunyiza moja kwa moja kwenye kilele jioni, na fuata maelekezo ya lebo ya dawa kuhusu kipimo na muda wa kusubiri kabla ya kuvuna. Thibitisha na afisa wa ugani wa eneo lako kwamba dawa hiyo imesajiliwa nchini mwako."}

### Content rules

1. **Real East African agriculture.** Name real crops, pests, diseases,
   regions and seasons. Use msimu wa mvua ndefu (long rains) and mvua fupi
   (short rains), not European seasons.
2. **Use the vocabulary farmers use.** Anchor terms: kilimo (agriculture),
   mkulima (farmer), shamba (farm), udongo (soil), mbegu (seed), mbolea
   (fertiliser), samadi (manure), wadudu (pests), magonjwa (diseases), kupanda
   (to plant), kupalilia (to weed), mavuno (harvest), kuvuna (to harvest),
   umwagiliaji (irrigation), mahindi (maize), maharagwe (beans), mihogo
   (cassava), viazi (potatoes), viazi vitamu (sweet potatoes), mtama
   (sorghum), ulezi (millet), ndizi (banana), kahawa (coffee), chai (tea),
   nyanya (tomato), vitunguu (onion), ng'ombe (cattle), mbuzi (goat), kuku
   (chicken), viwavijeshi (armyworm), kutu (rust), ukungu (blight/mildew).
3. **Never invent pesticide doses or product names.** Describe the active
   ingredient and the principle, then say the farmer must follow the label and
   confirm with the local extension officer, because registrations differ by
   country. Mention protective clothing and the pre-harvest waiting period.
4. **Never invent crop variety names.** Use real ones — Shangi, Tigoni, Kenya
   Mpya for potato; Rosecoco, Mwitemania for beans; H614, DK8031, Duma 43 for
   maize. If unsure, say to ask the local seed stockist or extension office.
5. **No filler openers.** Never start with "Bila shaka", "Hakika", "Ndio",
   "Certainly", "Sure". Start on the substance.
6. Never write "Mimi ni AI" or "As an AI".

### Answer length

- **30% LONG (150–300 words)** — a scenario with diagnosis and a practical plan
- **50% MEDIUM (50–120 words)** — one concept explained with a concrete number
- **20% SHORT (10–40 words)** — a direct factual answer

---

## BATCH 1 — Swahili questions, Swahili answers (1,200 entries)

> Generate 250 entries. Question in Swahili, answer in Swahili.
> Cover: mahindi na viwavijeshi, maharagwe, mihogo, viazi, ndizi, kahawa,
> udongo na mbolea, maji na umwagiliaji, misimu ya mvua, uhifadhi wa mazao
> baada ya mavuno, mifugo, kuku.
> Follow the system prompt exactly. JSONL only, first character `{`.
> Do not repeat any question already produced in this conversation.

Repeat with "endelea, 250 zaidi, usirudie chochote hapo juu" until you reach 1,200.

---

## BATCH 2 — cross-language (600 entries)

This is what makes the capability verifiable. A judge will very likely ask an
English question and request a Swahili answer, or ask in Swahili and expect the
model to cope. Train both directions explicitly.

> Generate 250 entries mixing these four patterns evenly:
>
> 1. English question, Swahili answer — where the question explicitly asks for
>    Swahili. e.g. "Explain in Swahili why crop rotation improves soil fertility."
> 2. Swahili question, Swahili answer, where the question asks for simple
>    language. e.g. "Nieleze kwa lugha rahisi kwa nini mzunguko wa mazao ni muhimu."
> 3. Swahili question, English answer — where the question asks for English.
>    e.g. "Naomba unieleze kwa Kiingereza jinsi ya kudhibiti viwavijeshi."
> 4. Requests to translate or explain an agricultural term across the two
>    languages. e.g. "What is 'viwavijeshi' in English, and how do I control it?"
>
> Follow the system prompt exactly. JSONL only, first character `{`.

---

## BATCH 3 — terminology and extension practice (700 entries)

> Generate 250 entries covering agricultural terminology in both languages and
> how an extension officer explains technical ideas to a farmer who may not read.
>
> Include: paired glossary entries (Swahili term, English term, what it means
> in practice), explanations of fertiliser grades and rates in Swahili, how to
> describe symptoms of the main crop diseases in Swahili, safety instructions
> for handling agrochemicals in Swahili, and market and cost questions in
> Swahili using shillings.
>
> Follow the system prompt exactly. JSONL only, first character `{`.

---

## After generating

```bash
python3 train/validate_african.py
python3 train/prepare_data.py --skip-unverified
grep -c -i "mahindi\|mkulima\|viwavijeshi\|shamba\|mbolea" train/data/train.jsonl
```

That last count must be in the thousands. If it is near zero the English-only
filter is still rejecting the Swahili rows and there is no point retraining.
