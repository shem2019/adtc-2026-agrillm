# Correction batches — targeted at observed failures

The model was tested interactively and produced these specific errors. Each batch
below targets one of them. Run these AFTER the main 15 topics, save with:

    ./train/save_batch.sh <slug>

Observed failures, for reference:

| Failure | What the model said | Reality |
|---|---|---|
| Invented variety name | "'Karamoja' variety" for potato AND onion | Karamoja is a region of Uganda, not a variety |
| Seed rate off by ~200x | "5 kg seed potatoes per acre" | 800-1,200 kg/acre (seed potatoes are tubers) |
| Wrong operation | "thin the potato seedlings to 15 cm" | Potatoes are not thinned |
| Dangerous post-harvest | "dry tubers in the sun until brittle" | Cure in shade; sun greens them and forms solanine |
| Wrong climate | "Nakuru, where the climate is hot and dry" | Nakuru is ~1,850 m, temperate highland |

---

## BATCH A — real varieties (use slug: potato-horticulture)

> Generate 250 entries about SPECIFIC, REAL crop varieties grown in East Africa.
> Follow the system prompt exactly. JSONL only, first character `{`.
>
> Cover named varieties with their actual characteristics — maturity days,
> altitude range, disease resistance, typical yield, and what they are used for.
> Include at minimum:
>
> - Potato: Shangi, Tigoni, Kenya Mpya, Dutch Robijn, Sherekea, Unica, Asante,
>   Kabale, Victoria, Rwangume
> - Onion: Red Creole, Bombay Red, Jambar F1, Red Passion, Texas Grano
> - Maize: H614, H628, H513, DK8031, Duma 43, WE1101, Longe 5, SC Duma
> - Beans: Rosecoco (GLP2), Mwitemania, Wairimu, KAT B1, Nyayo, Chelalang
> - Tomato: Rio Grande, Cal J, Anna F1, Kilele F1, Assila F1
> - Cassava: MM96/4271, Nase 14, TME 419, Migyera, Serere
> - Banana: FHIA-17, FHIA-25, Gonja, Mpologoma, Kiganda
> - Sweet potato: Kabode, Vita, Ejumula, SPK004
>
> CRITICAL: only name varieties you are confident actually exist and are grown in
> East Africa. If you are not certain about a variety's characteristics, write an
> entry that says which variety to ask the local extension office or seed
> stockist about, rather than inventing figures. An invented variety name is the
> single worst failure this dataset exists to fix.
>
> Include at least 30 entries of the form "which variety should I plant for
> {crop} in {specific region/altitude}" answered with real named varieties.

---

## BATCH B — corrected potato and horticulture agronomy (slug: potato-horticulture)

> Generate 200 entries on potato, onion and tomato agronomy in East Africa,
> covering the full cycle: land preparation, seed rate, spacing, planting depth,
> hilling/earthing up, fertiliser, irrigation, pest and disease control,
> maturity indicators, harvesting, curing and storage.
>
> Be precise and correct on these points specifically, as a previous model got
> them badly wrong:
>
> - Seed potato rate is roughly 800-1,200 kg per acre (they are tubers, not
>   grain). State the rate in both kg/acre and kg/ha.
> - Potatoes are NOT thinned after emergence. They are earthed up / hilled.
> - Potatoes are cured in a shaded, humid, well-ventilated place for 10-14 days,
>   NOT dried in the sun. Sun exposure greens tubers and produces solanine,
>   which is toxic.
> - Maturity is judged by haulm senescence and skin set, not tuber length.
> - Ware potatoes are stored in the dark; seed potatoes may be green-sprouted in
>   diffused light, which is a different thing and worth explaining.
>
> Include several entries that explicitly correct these misconceptions.

---

## BATCH C — regional climate and altitude (slug: climate-seasons)

> Generate 200 entries on the actual agro-ecological conditions of specific East
> African locations, and what to grow there.
>
> Be accurate about altitude and climate. For example Nakuru sits at roughly
> 1,850 m and is a temperate highland, not hot and dry; Kitale is a high-altitude
> maize zone; Mombasa is coastal humid; Turkana and Garissa are arid.
>
> Cover: Nakuru, Kitale, Eldoret, Nyeri, Meru, Embu, Machakos, Kisumu, Kakamega,
> Mombasa, Turkana, Garissa, Arusha, Mbeya, Iringa, Dodoma, Morogoro, Kampala,
> Mbale, Kabale, Kigali, Musanze.
>
> For each, give altitude band, rainfall pattern and season names, main soil
> type, and which crops and varieties suit it. Include questions of the form
> "I farm near {place} at about {altitude} — what should I plant and when?"

---

## BATCH D — refusing to invent (slug: economics-extension)

> Generate 150 entries where the CORRECT answer is to decline to give a specific
> figure and say what to check instead.
>
> These teach the model to stop inventing. Cover cases where a responsible
> extension officer would not guess: exact pesticide dose without the product
> label, fertiliser rate without a soil test, variety recommendation without
> knowing the altitude and market, spraying interval without knowing the product,
> and market prices which change constantly.
>
> Each answer should: name what is missing, explain briefly why it matters, and
> say exactly who to ask or what to check. Keep them 40-120 words. Do NOT use
> the phrase "cannot be determined" — vary the wording naturally.
>
> Example shape:
> {"question": "How much Imidacloprid should I spray on my tomatoes?", "answer": "The rate depends on the specific registered product, its concentration, and your target pest, and applying the wrong rate is both ineffective and unsafe. Read the rate table on the product label, and confirm with your local agrodealer or extension officer that the product is registered for tomatoes in your country. Note the pre-harvest interval on the label and do not harvest before it has passed. Wear gloves, a mask and long sleeves when mixing and spraying."}

---

## After generating

```bash
python3 train/validate_african.py
python3 train/prepare_data.py --skip-unverified
grep -c -i "shangi\|tigoni\|kenya mpya\|rosecoco\|red creole" train/data/train.jsonl
```

That last count confirms real variety names reached the training data. If it is
near zero the batches did not survive filtering and there is no point retraining.
