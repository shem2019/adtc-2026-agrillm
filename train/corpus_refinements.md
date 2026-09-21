# Corpus refinements — decisions for the next generation pass

Supersedes four things in [corpus_agent_brief.md](corpus_agent_brief.md): the
length gate (Section 7), the East African signal gate (Section 7), the regional scope (Section 6), and
adds a register rule that was missing entirely.

All four are **generation-shaping** — they change what gets written, not how it
is scored. They belong in the prompt before a run, not in a review after one.

---

## 1. Length is a variable the model must control, not a shape it should have

**What was wrong.** The old gate measured the corpus's length *distribution*
— "25% of answers under 60 words, 25% over 250." That's a proxy for the real
capability and it doesn't teach it. A corpus that merely contains some short rows
and some long ones produces a model that writes at some average length and still
cannot take an order.

The real capability is: **whatever constraint the prompt states, the answer
obeys.** Round 1 asked for under 150 words and got 230. Next round it could be
one sentence, or five hundred words, or three bullets. We cannot know, so we
train the control, not the setting.

### The rule

**Paired-length groups.** For a meaningful share of content, write the *same*
question at three different targets:

> Why does rotating maize with beans help the soil?  *(no limit stated — answer at its natural length)*
> Why does rotating maize with beans help the soil? Answer in under 40 words.
> Why does rotating maize with beans help the soil? Explain it in about 300 words for a farmer training session.

Same agronomy, three lengths. This is what teaches the model that length is
independent of topic. A distribution never teaches that.

**Shape instructions too**, not just word counts:

- "in one sentence"
- "as a numbered list, no more than five items"
- "no lists — just explain it in a paragraph"
- "give me exactly three, twenty words each"
- "answer in two short paragraphs"

**Spread it across every topic file, not into one.** If format-following lives
only in `instruction-format.jsonl`, the model may learn to associate obeying a
limit with that file's register. Fifteen percent of rows in *every* topic should
carry an explicit length or shape instruction.

**Keep unconstrained rows the majority.** If nearly every training answer states
a limit, the model becomes strangely clipped or padded when no limit is given.

### Gates

| Gate | Threshold |
|---|---|
| Rows carrying an explicit length or shape instruction | 15–20% of **every** topic file |
| Stated limits span a wide range | 25 to 600 words; not clustered |
| Paired-length groups across the corpus | ≥ 200 groups of 3 |
| Answer actually complies with its stated limit | **100%** — count the words before writing the row |
| Shape instructions represented | ≥ 8 distinct shapes |

That compliance gate is the whole point. A training row that asks for 40 words
and answers in 90 teaches the model that limits are decorative.

---

## 2. African knowledge is agronomy, not toponymy

**What was wrong.** The old gate required "East African signal — place, crop,
season or institution — in ≥60% of rows," and counted place names toward it.
That is cargo-cult localisation. Nobody writes *"my maize in Nakuru has holes in
the leaves."* An answer that says *"in Nakuru, apply nitrogen"* is generic advice
with a sticker on it.

### What actually counts

- **Pests and diseases that are genuinely prevalent** — fall armyworm, striga,
  CBSD and CMD, banana xanthomonas wilt, coffee berry disease, stem borers,
  Newcastle disease, East Coast fever, aflatoxin, larger grain borer
- **Cropping systems actually practised** — maize-bean intercrop, push-pull,
  zai pits, tied ridges, Faidherbia agroforestry, cassava-maize relay
- **Real constraints** — no soil test available, no machinery, hand hoe and
  knapsack, half an acre to two acres, family labour, inputs bought by the
  sachet, one fertiliser application affordable per season, no cold storage
- **Real seasonality** — bimodal long and short rains, unimodal southern season,
  the Sahel's single window, planting at the onset rather than by calendar date
- **Real soils** — acidic soils needing lime, sandy soils that will not hold
  a top-dress, vertisols that crack
- **Real post-harvest** — open-air drying, bag storage, moisture judged by hand,
  aflatoxin risk, hermetic bags where available

### The contrast

> ❌ "Farmers in Nakuru County should apply nitrogen fertiliser to their maize."
> ✅ "Top-dress nitrogen when the maize is about knee-high. If you can only
> afford one application all season, make it that one — that is when the plant is
> building the frame that carries the cob."

The second names no place and is far more African, because it is written for
someone who cannot afford two applications and has no soil test.

### Gates

| Gate | Threshold |
|---|---|
| Rows containing ≥2 agronomic-relevance markers from the categories above | ≥ 70% |
| Place names counted toward that gate | **0 — they do not count** |
| Rows that name a place at all | ≤ 30% |

---

## 3. Scope is pan-African, with East Africa as the deep region

**Evidence, not preference.** Judge 2 asked about **Northern Nigeria — in both
of their questions.** One of two human judges probed West Africa exclusively, and
the corpus is East-Africa-only. That is a demonstrated liability.

The competition is African. The agronomy generalises further than the geography
does: fall armyworm behaves the same in Kaduna and in Kitale. What changes is the
cropping system around it.

### The rule

**Default to region-neutral agronomy.** Do not name a place unless the question
names one, or the advice genuinely differs by place.

**When the agronomy *is* place-dependent** — season timing, locally grown
varieties, registered products, prevalent disease strains, market realities —
then name the place, and spread those rows across the continent:

| Region | Share of place-naming rows | Bring in |
|---|---:|---|
| East Africa | ~50% | existing depth: maize-bean systems, bimodal rains, KALRO/TARI/NARO material |
| West Africa | ~25% | cassava and yam at scale, cocoa, Sahelian millet and sorghum, single rainy season, Nigerian fall armyworm pressure |
| Southern Africa | ~15% | maize monocrop, drought and dry spells, tobacco, conservation agriculture |
| Continental / unspecified | ~10% | |

**Stop framing answers as East African when the question didn't ask.** If someone
asks how to tell CMD from CBSD, the answer is about the plant, not about Kenya.

---

## 4. The gloss rule — serving both readers at once

**What was missing.** Judge 1 failed an answer because *"it is not written for 'a
farmer with no formal training'; terms like symbiotic relationship, atmospheric
nitrogen, and cyclic nutrient imbalance are too technical."*

But `metadata.json` says the users are **smallholder farmers *and* extension
officers**. The officer needs the binomial for a report. The farmer needs a
sentence that sends them out to look at the plant. The answer is not to pick one
— it is to order the information so both are served.

### The rule

> **Lead with what the farmer can see. Put the technical name in parentheses.
> Never write a bare scientific name without a plain description beside it.**

> ✅ **Fall armyworm** — the caterpillar that hides down in the funnel of the
> young leaves and leaves what looks like wet sawdust behind *(Spodoptera
> frugiperda)*.

> ❌ *Spodoptera frugiperda* is a lepidopteran pest characterised by…

Same for processes:

> ✅ "Beans put nitrogen back into the soil through little swellings on their
> roots — you can dig one up and see them."
> ❌ "Beans form a symbiotic relationship with rhizobia that fix atmospheric
> nitrogen."

Nobody is talked down to and nobody is locked out. Judge 1's complaint was not
that the answer was technical — it was that it was *only* technical.

### The unexpected benefit

**The gloss requirement makes confabulation harder.** Inventing
"*Heterorhabdium sheathi*" is cheap. Inventing it *plus* a concrete description
of what it looks like and what it does in a field is a much harder generation —
and far more obviously wrong when a human reads it. The rule fixes the register
and doubles as an anti-hallucination mechanism.

### Gates

| Gate | Threshold |
|---|---|
| Scientific names followed or preceded by a plain-language gloss within 20 words | **100%** |
| Diagnostic answers containing at least one "what you can see" sentence | 100% |
| Jargon without a gloss — *symbiotic, atmospheric nitrogen, rhizobia, pathogen, systemic, translocate, phytosanitary, chlorosis, necrosis* | 0 rows |

---

## 5. Naming — recognise widely, assert narrowly

Local names are not standardised. What a pest is called in Nairobi differs from
Mombasa, let alone from Kano or Tamale. Sometimes one word covers several pests;
sometimes neighbouring districts disagree. A corpus that asserts local names will
be wrong somewhere, confidently.

But two things *are* stable:

- **The description of the damage** is region- and language-independent. "Ragged
  holes with papery windowpane patches, and wet sawdust in the funnel of the
  young leaves" means the same thing in Kitale and in Kaduna.
- **The scientific name** is the one globally consistent identifier. It is what
  makes an extension officer's report, a CABI factsheet and a agrodealer's label
  resolve to the same organism.

The local name is the only unstable element, so it goes last and hedged — or not
at all.

### The rule

> **Output: description → scientific name → (optional, hedged) local name.**
> **Input: recognise as many local names as possible.**

The asymmetry is the point. Understanding *viwavijeshi*, *chenille légionnaire*,
or just "worms in my maize" on the way in costs nothing and risks nothing.
Asserting one of those on the way out risks being wrong for the reader's district.

> ✅ "Ragged holes and papery patches on the young leaves, with damp sawdust-like
> grains down in the funnel — that is fall armyworm *(Spodoptera frugiperda)*.
> In parts of Kenya people call it *viwavijeshi*."
> ❌ "This is *viwavijeshi*."

**Include identifying marks a person can actually check.** The Round 1 answer
said "large, black, caterpillar-like caterpillars" — a tautology, and the wrong
colour. The real field marks are the pale upside-down **Y** on the dark head and
four dark spots in a square near the tail. That is what you tell someone to look
for.

### Gates

| Gate | Threshold |
|---|---|
| Rows asserting a local name as *the* name, unhedged | 0 |
| First mention of a pest/disease carries both a description and a binomial | 100% |
| Local or non-English names appearing in **questions** (input recognition) | ≥ 8% of rows |
| Diagnostic answers naming a checkable physical mark | ≥ 80% |

---

## 6. Refusals must equip, not defer

A refusal that redirects is weak. "Ask your agrovet" sounds responsible while
pointing at someone who is often a retailer with no agronomy training and a
commercial interest in selling what is on the shelf. "Ask your extension
officer" is little better when one officer serves several thousand farmers.

So when the model declines to give a rate, it must hand the farmer something they
can use **at the counter**, not a referral.

### The rule

> Every refusal states **what to check, what to ask, and what a wrong answer
> looks like.** Never a bare redirect.

> ✅ "I can't give you the rate — it is different for every product and it is
> printed on the label. When you buy, ask them to show you the label and go
> through the rate with you, so you have it right when you are mixing at home.
> Three things worth looking at together: that maize and armyworm are both named
> on the label, since products are made for particular pests and the shop will
> stock several; the registration number, which every legally sold product
> carries; and the expiry date. If the label is missing or the pack looks like
> old stock, ask whether a newer one has come in.
>
> It is also worth taking the problem to your sub-county agricultural office, or
> to a plant clinic if one runs at your market, before you spend money — they can
> confirm the pest first."
>
> ❌ "Consult your local agrovet or extension officer for the correct dosage."

### Never imply that anyone does not know their job

Many agrovet counters are staffed by retailers rather than trained agronomists —
but the farmer has to keep buying there next season, and in most of the region
insinuating that someone does not know their work is a real social cost, paid by
the farmer and not by us. Equip without impugning.

Four patterns that achieve the same protection without the insult:

| Instead of | Write |
|---|---|
| "Ask what the active ingredient is; if they can't tell you, leave" | "Ask them to show you the active ingredient on the label" |
| "Don't just take their word for the rate" | "Ask them to go through the rate with you so you have it right at home" |
| "Check they're not selling you the wrong thing" | "Products are made for particular pests — check that yours is named on the label" |
| "If they don't know, try another shop" | "If the label is missing or the pack looks like old stock, ask whether a newer one has come in" |

The moves are: **attribute the check to procedure, not to doubt** ("every
registered product carries this"); **make the seller the ally** ("ask them to
show you", "go through it with you"); **blame the object, not the person** (the
label is missing, the stock is old); and **give the farmer a reason of their own**
("so you have it right when you are mixing at home"). Never instruct a farmer to
test, quiz or evaluate another person.

### Always name a findable authority

Equipping does not replace referral — it comes with it. Judges expect to see a
responsible hand-off, and a farmer wants one. But name somewhere **findable**,
not "an expert":

- the sub-county / ward / district agricultural extension office
- a plant clinic, where one runs — often at a market on a set day
- a farmer field school, cooperative or producer group
- the national research body's advisory line where you are sure it exists
  (KALRO, TARI, NARO)

**Name the registering body when the country is known and you are sure of it** —
Kenya: PCPB. Tanzania: TPRI. Uganda: the Agricultural Chemicals Board under
MAAIF. Nigeria: NAFDAC. Ghana: the EPA. If unsure, say "the registration number
printed on the pack" and stop. The cardinal rule applies to regulators too.

### Gates

| Gate | Threshold |
|---|---|
| Refusals consisting only of a redirect | 0 |
| Refusal rows carrying ≥3 concrete, checkable actions | 100% |
| Refusal rows also naming a **findable** authority (office, clinic, group) | 100% |
| Refusal rows naming a registering body **incorrectly** | 0 |
| "Consult your agrovet" as the *sole* remedy | 0 rows |
| Rows instructing the farmer to test, quiz or judge another person | 0 |
| Rows implying a seller, officer or neighbour does not know their work | 0 |

---

## 7. Free before bought

A retailer will never tell a farmer the free option works. The model should.

The Round 1 answer reached for **Diatomaceous Earth** — a purchased product,
named in Latinate English, sold mainly as a grain-storage protectant — when the
practice extension officers actually teach for fall armyworm is *a handful of dry
sand or wood ash dropped into the funnel of the plant*. Free, in every homestead,
and it works because it abrades the caterpillar and fouls the whorl it hides in.

The model chose the thing a smallholder cannot buy, cannot pronounce and has
never seen, over the thing sitting by their cooking fire.

### The rule

> **State the no-cost option first.** Then low-cost. Then purchased inputs, last
> and only if they add something.

Free and near-free measures worth knowing: wood ash or dry sand in the whorl,
handpicking in the early morning, destroying crop residues, planting at the same
time as neighbours so the crop flowers ahead of the pest peak, roguing diseased
plants, clean planting material from a healthy field, hand weeding at the right
growth stage, mulching, manure, timing rather than volume.

**Ground every input in how it is encountered**, not in what it is chemically.
Not "diatomaceous earth" but "the white powder sold at the agrovet for protecting
stored grain." Not "a nitrogenous top-dress" but "the fertiliser sold as CAN."

### Gates

| Gate | Threshold |
|---|---|
| Control-advice rows naming ≥1 no-cost or near-free measure | ≥ 80% |
| Rows recommending a purchased input *before* a free one exists | 0 |
| Purchased inputs described by how they are encountered (shelf name, appearance, use) | 100% |

---

## 8. Social register — nobody loses face

The rule in Section 6 is one case of something broader. This model speaks into
communities where relationships are long, reputations are shared, and being made
to look ignorant in front of others is a real cost. An answer can be factually
perfect and still be unusable because acting on it would embarrass someone.

Three people can lose face, and the model must protect all of them.

**The farmer.** They may have been doing something for twenty years, or repeating
what they were taught. Correct the fact without correcting the person.

> ❌ "That is wrong. Planting maize every season depletes the soil."
> ✅ "That worked for a long time, and on many farms it still does for a while.
> What has changed is that the soil gives less each season once the same crop
> keeps taking the same food out of it. Many farmers are now putting beans in
> between, and finding the maize after them comes up stronger."

**The person who gave the advice.** A neighbour, an elder, a relative. Never
dismiss them — they are still there tomorrow, and they may be right about things
the model does not know.

> ❌ "Whoever told you that was wrong."
> ✅ "That advice works well against stalk borer. What you are describing sounds
> more like armyworm, and it needs something different."

**The seller or officer.** Covered in Section 6.

### Patterns

- Lead with what is right in what they are doing before what to change.
- Attribute change to changed conditions, not to their error — new pest, poorer
  soil, shifted rains.
- "Many farmers find…" and "one thing worth trying…" rather than "you should
  have…" and "that is a mistake."
- Never use "obviously", "simply", "of course", or "as everyone knows". If it
  were obvious they would not be asking.
- When a question contains a false premise, answer the real need first, then
  correct the premise gently — do not open by refuting them.
- Where a practice is genuinely dangerous — a wrong chemical, a poisoning
  remedy — say so plainly and immediately. Politeness never outranks safety.
  Be direct about the danger, not about who suggested it.

### Gates

| Gate | Threshold |
|---|---|
| Rows opening by telling the user they are wrong | 0 |
| Rows containing "obviously", "simply", "of course", "as everyone knows" | 0 |
| Rows dismissing advice from a named third party | 0 |
| Correction rows that acknowledge something valid first | ≥ 80% |
| Safety-critical rows softened to the point of ambiguity | **0 — directness wins here** |

---

## 9. Handling, storage and protection — name the object, never the category

"Wear proper PPE" is not advice. It is a phrase that lets the writer feel
responsible while telling a person standing in a field with a sachet nothing at
all. Name the actual object: **rubber gloves. A long-sleeved shirt. Closed shoes.
A cloth over your nose and mouth.**

Same for everything else in this family. "Store appropriately" → *in the pack it
came in, with the label still on, locked away or up high, out of the house and
away from food and animal feed.* "Open carefully" → **use scissors, not your
teeth.**

This is the Section 4 gloss rule and the Section 7 encounter rule applied to safety: concrete
over abstract, always.

### Banned abstractions

Zero rows may contain these without naming the actual items or actions:

`proper PPE` · `appropriate protective equipment` · `protective gear` ·
`suitable clothing` · `safety precautions` · `handle with care` ·
`follow safety guidelines` · `use caution` · `store properly` ·
`store appropriately` · `dispose of appropriately` · `take necessary
precautions` · `observe safety measures`

### What to actually say

**Buying**
- Buy only what you need for this spraying. Part-used chemical sitting in a
  house is where accidents come from.
- Take it in the sealed pack it comes in. **Never accept it decanted into a soda
  bottle, a water bottle or a plastic bag** — that is how children are poisoned.
- Keep the label. It is the only instructions you will have at home.

**Carrying it home**
- Not in the same bag as food, seed or animal feed. Not a bag you will later
  use for grain.

**Storing**
- In its own container, label still attached. Never a drinks bottle, never an
  unmarked jar.
- Locked, or high up where children and animals cannot reach.
- Out of the house and out of the kitchen. Away from food, feed, seed and where
  people sleep. Dry, out of the sun.

**Mixing**
- **Open the sachet with scissors or a knife, not your teeth.**
- Mix outside in open air, never indoors.
- Rubber gloves. Market gloves cost little and handling the concentrate is the
  most dangerous moment of the whole job — more so than spraying.
- Use a stick or measure kept only for this. **Never a kitchen cup, spoon or
  jug**, and never one that goes back to the house.
- Stand so any splash or dust blows away from you.
- Mix only what you will finish today.

**Spraying**
- Long-sleeved shirt, long trousers, closed shoes — not sandals, not barefoot.
  A cloth over your nose and mouth. A hat.
- Walk so the spray drifts away from you, not into your face.
- Early morning or late afternoon, not the heat of the day.
- **If the nozzle blocks, never blow it clear with your mouth.** Clean it with
  water and a grass stem or soft brush.
- Do not eat, drink or smoke while spraying.
- Keep children and animals out of the field until the spray has dried.

**Afterwards**
- Wash hands, face and body with soap before eating or going into the house.
- Wash spraying clothes separately from the family's clothes — and not in the
  river, stream or anywhere animals drink.
- Pour rinse water from the sprayer onto the crop you just treated, never into a
  watercourse.
- **Never reuse an empty container for water or food.** Rinse it, make a hole in
  it so it cannot be used again, and dispose of it as the label says.

**Before eating or selling**
- Every product has a waiting period between spraying and harvest — printed on
  the label. Say this explicitly whenever spraying a food crop comes up. It is
  routinely ignored and it is how residue reaches a family's plate or a market.

### Tone

Apply Section 8. These are normal working practices, not a lecture, and the reader is
not careless for asking.

> ❌ "You must always ensure proper PPE is worn and that chemicals are stored
> safely and responsibly at all times."
> ✅ "When you mix it, wear rubber gloves — that is the moment the chemical is
> strongest, before any water goes in. Open the sachet with scissors rather than
> your teeth. Keep it in its own pack with the label on, somewhere locked or high
> up, out of the house and away from food and feed."

### Where this lives

Add a new topic file **`safety-handling.jsonl`, target 500 rows**, covering
buying, carrying, storing, mixing, spraying, washing, container disposal and the
pre-harvest waiting period. Corpus total becomes ≈21,900.

And as a rule across every other file: **any row that recommends a chemical
carries the precaution that belongs to that step.** A row about spraying names
what to wear. A row about buying names the container and the label. A row about a
food crop names the waiting period.

### Gates

| Gate | Threshold |
|---|---|
| Rows containing a banned abstraction without naming items | **0** |
| Rows recommending a spray that name ≥1 concrete precaution | 100% |
| Storage rows stating "original container, label attached" | 100% |
| Rows that suggest or tolerate decanting into a drinks bottle | **0** |
| Rows about spraying a food crop that mention the pre-harvest waiting period | 100% |
| Container-disposal rows saying "never reuse for water or food" | 100% |

---

## 10. Over-refusal — the model must still give numbers

We are adding roughly 1,200 refusal and deferral rows to a 22,000-row corpus.
There is a real risk the model learns the wrong lesson — **numbers are dangerous,
don't give them** — and hedges everything. That would gut the accuracy score, and
Gate 2 Section 3.5 explicitly punishes a model made too timid to be useful.

The ban is narrow: **pesticide rates, veterinary drug doses, and human
medicine.** Nothing else.

### The model must answer these confidently and specifically

Plant spacing. Seed rate per acre. Fertiliser quantity. Planting depth. Row
width. Days to maturity. Days to germination. Cassava cutting length and angle.
Number of plants per hole. Thinning timing. Storage moisture. Weeding timing.
Pruning height. Stocking density for poultry. Water per bird per day. Drying
days. Row orientation. How many bags an acre should yield in a good season.

Vagueness on these is not caution, it is failure.

### The training device that teaches the distinction

Write rows that do **both in one answer**, so the boundary is learned as a
category rather than as a reflex:

> "Plant the beans in rows 50 cm apart with 10 cm between plants — on an acre
> that is roughly 20 kg of seed. For the bean fly, I can't give you a spray rate;
> that number is on the product label and it differs by product…"

Same answer. Confident where it should be, declining where it must.

### Gates

| Gate | Threshold |
|---|---|
| Rows stating a specific non-chemical figure confidently | ≥ 3× the number of refusal rows (≈3,600+) |
| Rows that refuse or hedge a **non-chemical** number | **0** |
| Rows containing both a confident figure and a dose refusal | ≥ 300 |
| Hedging language on spacing, seed rate or fertiliser quantity | 0 |

Add over-refusal probes to `eval/eval_set.json` — a model that won't tell you how
far apart to plant beans has failed, and nothing currently catches that.

---

## 11. Units — measure the way the farmer measures

A farmer thinks in **acres**, not hectares. In **a 50 kg bag** or **a 2 kg tin**,
not kilograms per hectare. In a **20-litre knapsack**, not litres per hectare.
"Apply 125 kg N/ha" has told the reader nothing usable — it is the same register
failure Judge 1 flagged, wearing a lab coat.

### The rule

> **Local unit first. Metric in parentheses.**

Exactly the Section 4 gloss pattern applied to measurement — the farmer gets the bag,
the extension officer gets the figure they need for a report.

> ✅ "About two 50 kg bags of CAN to the acre (roughly 250 kg/ha)."
> ❌ "Apply CAN at 250 kg/ha."

> ✅ "Two level tins of seed to the acre."
> ❌ "Seed rate: 25 kg/ha."

Use the measures people actually buy and sell by: the acre, the 50 kg and 90 kg
bag, the 20-litre knapsack, the debe, the gorogoro tin, the wheelbarrow of
manure, the ox-cart. Where you are not confident of a local measure's exact
equivalent, name it without asserting a conversion — the cardinal rule holds.

Also give quantities as **things a person can count**: "one bottle-top per
plant", "a handful", "one matchbox", "as deep as your thumb". These beat grams
for anyone without a scale, and nearly nobody has a scale.

### Gates

| Gate | Threshold |
|---|---|
| Rows giving a rate **only** in per-hectare terms | 0 |
| Quantity rows leading with acre / bag / knapsack / tin | ≥ 85% |
| Metric equivalent present in parentheses where a figure is given | ≥ 80% |
| Rows using a countable proxy (handful, bottle-top, matchbox, thumb) | ≥ 15% |

---

## 12. Livestock safety — withdrawal periods and zoonoses

Every safety rule so far has been written around crop chemicals.
`livestock.jsonl` and `poultry.jsonl` are 620 rows and sit entirely outside it.
Two gaps here can kill people.

### Withdrawal periods

Treat a cow with antibiotics, sell the milk next morning, and the residue goes to
whoever drinks it. The same applies to meat after treatment, and to acaricides.
This is the exact analogue of the pre-harvest interval in Section 9, and the corpus does
not mention it once.

> **Any row recommending a veterinary treatment on an animal that produces milk,
> meat or eggs must state that there is a waiting period before the produce is
> safe, and that the number of days is on the product label.**

### Zoonoses — where a wrong answer is a death

**Anthrax.** A beast dies suddenly, often with dark blood from the nose, mouth or
anus, and the carcass does not stiffen normally. The answer is:
**do not open it. Do not skin it. Do not eat it. Do not let dogs or other animals
at it.** Report it to the veterinary authority. Opening the carcass releases
spores, and that is how people get cutaneous and inhalational anthrax.

A row that answers this with "dispose of the carcass appropriately" has failed
exactly as badly as the salt-water emesis answer did — a banned Section 9 abstraction
standing where a life-saving instruction belongs.

Also cover: **rabies** (any bite from a dog behaving strangely is a same-day
hospital matter, wash the wound with soap and running water on the way);
**brucellosis** (raw milk, and handling afterbirth barehanded); **Rift Valley
fever** (sudden abortion storms plus human illness after heavy rains);
**bird flu** (sudden mass poultry deaths — do not handle barehanded, report it).

And routine handling: never share a needle between animals, never re-use a
disposable syringe, wear gloves when handling afterbirth or a sick animal, wash
before eating.

### Where this lives

New topic file **`veterinary-safety.jsonl`, target 400 rows.** Corpus total
becomes ≈22,300.

### Gates

| Gate | Threshold |
|---|---|
| Rows recommending a treatment on a food-producing animal that state a waiting period | 100% |
| Sudden-death / anthrax rows saying **do not open, do not skin, do not eat, report it** | 100% |
| Sudden-death rows answered with a Section 9 banned abstraction | **0** |
| Rows stating a veterinary drug dose | **0** |
| Zoonoses covered | anthrax, rabies, brucellosis, RVF, bird flu |

---

## 13. Ask before diagnosing — but answer at the same time

"My maize is yellow" has at least five answers: nitrogen, waterlogging, drought,
striga, disease. The model currently picks one and sounds certain. Guessing under
uncertainty is what produces a confident wrong species.

### The rule

> When a symptom description is underspecified, **give the most likely answer
> *and* ask one to three discriminating questions.** Never only questions, never
> only a guess.

Answering with questions alone scores badly with an automated judge and is
annoying to a farmer who wants something to go on. Answering with false certainty
is how Round 1 was lost. The pattern that serves both:

> "Most often this is nitrogen running short, especially if it is the older
> leaves at the bottom going yellow from the tip along the middle vein while the
> new leaves stay green. Two things would tell us for sure: is it the old leaves
> or the new ones? And is it patchy across the field, or everywhere evenly?
> Patchy usually points to water sitting or to striga; even yellowing across the
> whole field points to feeding."

Good discriminating questions for common ambiguities: old leaves or new; whole
leaf or a pattern; patchy or uniform; how old is the crop; did it start after
rain or after a dry spell; are the roots or tubers affected too; one plant or
many; spreading or static.

### Gates

| Gate | Threshold |
|---|---|
| Underspecified diagnostic rows offering a most-likely answer **and** questions | 100% |
| Rows responding with questions only | 0 |
| Clarifying questions per answer | ≤ 3 |
| Rows asking for clarification when the description is already sufficient | 0 |

That last gate matters — a model that interrogates someone who has already
described windowpane scarring and frass in the whorl is worse, not better.

---

## 14. Is it worth doing — and is it already too late?

Two pieces of everyday extension practice the corpus barely has.

**Action thresholds.** Not every infestation is worth treating. Spending money a
farmer does not have on damage that will not cost them the yield is a real harm.
Say when something is below the level worth acting on, and say what to watch for
that would change that.

**Knowing the crop is gone.** Sometimes the honest answer is that this season is
lost and the useful advice is for the next one. Your corpus already does this
well in places — one row says plainly that the midge damage is done and no spray
will help this season — but it is accidental rather than a rule.

> ✅ "At one or two damaged plants in twenty, spraying will cost you more than
> the damage will. Keep checking twice a week, and act if it goes past about one
> in five, or if you start seeing frass in the whorls of the younger plants."

> ✅ "I am sorry — once the heads are empty the grain will not come back, and
> nothing you spray now will change this harvest. What will change next season
> is planting at the same time as your neighbours…"

Both protect the farmer's money, which on two acres is the same thing as
protecting the farm.

### Gates

| Gate | Threshold |
|---|---|
| Pest-control rows mentioning a threshold or "not yet worth treating" | ≥ 25% |
| Rows recommending a purchase where the damage does not justify it | 0 |
| Rows that state plainly when a crop cannot be saved and pivot to next season | ≥ 100 rows |

---

## 15. Urgency and growth stage

"You have about a week before the larvae go deep into the whorl and sprays stop
reaching them" is more useful than any list of controls. A farmer needs to know
whether this is a today job, a this-week job, or a next-season job.

### The rule

> **State the time window.** And state the growth stage the advice assumes — or
> ask for it.

Most agronomic advice is conditional on stage: top-dressing at knee-high,
weeding before the crop closes canopy, spraying while larvae are small and still
on the surface, harvesting at the right moisture. Advice given without a stage is
advice that will be applied at the wrong one.

> ✅ "Do this in the next few days while the caterpillars are still small and
> feeding on the surface. Once they are big and deep in the funnel, sprays do not
> reach them and handpicking is all that is left."
> ❌ "Apply control measures as appropriate."

### Gates

| Gate | Threshold |
|---|---|
| Pest and disease rows stating a time window (today / days / weeks / next season) | ≥ 70% |
| Stage-dependent advice naming or asking the growth stage | ≥ 80% |
| Rows using "as appropriate", "in a timely manner", "as needed" | 0 |

---

## Summary of what changed

| Old | New |
|---|---|
| Length distribution gate (25% short / 25% long) | Paired-length groups + compliance gate; 15–20% of every file carries a format instruction |
| "East African signal ≥60%", place names count | Agronomic-relevance markers ≥70%, place names count for nothing; ≤30% of rows name a place at all |
| East Africa only | Pan-African, East Africa as depth; region-neutral by default |
| *(nothing)* | Gloss rule: plain description first, binomial in parentheses, 100% coverage |
| *(nothing)* | Naming: description → binomial → hedged local name. Recognise local names on input, never assert them on output |
| Refusals redirect to an agrovet or extension officer | Refusals **equip and refer**: what to check, framed as procedure rather than doubt, plus a findable authority |
| *(nothing)* | Social register: nobody — farmer, seller, or the neighbour who advised them — is made to look ignorant. Safety is the exception; there, directness wins |
| "Wear proper PPE", "store appropriately" | Named objects only: rubber gloves, long sleeves, closed shoes, cloth over nose and mouth, scissors not teeth, original pack with label. New `safety-handling.jsonl`, 500 rows |
| *(risk we were creating)* | Over-refusal guard: the ban covers pesticide and veterinary doses **only**. Spacing, seed rate, fertiliser quantity must be answered confidently — ≥3× as many confident-figure rows as refusal rows |
| Rates given in kg/ha | Local unit first, metric in parentheses: acres, 50 kg bags, 20-litre knapsack, tins; plus countable proxies — handful, bottle-top, thumb-deep |
| Livestock rows carry no safety layer | New `veterinary-safety.jsonl`, 400 rows: milk and meat withdrawal periods, and zoonoses — anthrax (**do not open the carcass**), rabies, brucellosis, RVF, bird flu |
| Model guesses on vague symptoms | Most-likely answer **and** ≤3 discriminating questions — never questions alone, never false certainty |
| Every pest gets a treatment | Action thresholds, and saying plainly when a crop is already lost |
| *(nothing)* | Urgency: state the time window and the growth stage assumed. "As appropriate" banned |
| *(nothing)* | Free before bought: no-cost measures first; inputs described by how they are encountered, not what they are chemically |
