## Inspiration

My parents recently retired from teaching and started farming, and they quickly
found that farming runs on advice. Which seed suits this season, what is eating
the maize, whether a spray is safe: the questions came every week, and they often
did not know where to find the answers.

They relied on extension officers, and extension officers are scarce; one officer
serves thousands of farmers, so help rarely arrived when the question did. The
other source of advice was AI, and that meant a monthly subscription with no room
in the budget for it, internet that is unreliable and expensive at the farm, and
a machine good enough to run it, which they did not have.

That is the situation of millions of smallholders across Africa, and it gave me
the idea: a model that offers farming advice at no cost, on a budget machine,
with no internet needed once it is installed. When I read the ADTC brief, it
asked for exactly that: useful AI on the 8 GB laptop already on the desk,
working offline, with no subscription. The competition and the problem I wanted to
solve turned out to be the same thing, and that idea became AgriLLM.

## What it does

AgriLLM extends the reach of extension officers by answering everyday farm
questions on demand, offline, on a budget laptop. It speaks the way an officer
talks to a farmer: it names the weed, pest or disease, explains why it is
happening, and says what to do this week and whom to call, in plain words for
people with no technical background.

A farmer asks: *"Small purple flowering weeds are coming up around my maize in
western Kenya and the maize is stunted even though I applied fertiliser. What is
this and how do I control it?"* AgriLLM answers:

> *"These are Striga hermonthica, also known as witchweed, which is a parasitic
> weed. It attaches to maize roots underground and sucks water and nutrients from
> the plant, causing the stunting even with reasonable fertiliser application."*

In two sentences the farmer learns the weed's name, why the maize is stunted, and
why more fertiliser will not fix it. That is the knowledge needed to act, in words
anyone can follow. AgriLLM gives the same kind of answer across the farm:

- **Chickens dying with twisted necks:** it names Newcastle disease and tells
  the farmer to separate the sick birds and call an animal health worker.
- **Why rotate maize:** it explains in a few plain sentences that maize is a
  hungry crop and that beans give the soil a break and put goodness back.
- **A pesticide dose question:** it points to the product label and the
  agrodealer, because rates differ by country and a wrong rate can poison
  someone.

Fine-tuning made the difference. Against the model it was built on, AgriLLM
scores **three and a half times higher** on a 24-prompt behavioural test, gets
diagnosis and agronomy right **nearly five times as often**, gives plain-language
answers **up to eight times as often**, and cuts safety failures **from five to
one**.

## How it was built

**The base model was chosen to suit a farmer's connection and laptop.** Seven
candidates from 0.5B to 4B were measured with the official profiler.
Qwen2.5-1.5B answered level with the 3B and 4B models while running two to three
times faster, using half the memory and downloading at about half the size. For
a farmer, that means a quick download and fast answers, even on modest hardware.

**A corpus was written for African farms.** Round 1 trained on 18,248 mostly
third-party rows, and a judge found the result unfit for field use: it invented
doses and species. For Gate 2 the data was rebuilt from scratch as **6,703
verified rows** of African extension advice across 25 files, each checked by a
script that removes stated pesticide rates, vague safety advice and templated
duplicates. That corpus is what makes the advice safe, practical and local.

**Training ran in two stages, breadth then correctness.** Stage 1 uses the
third-party dataset AI71ai/agrillm-train-146k so the model recognises a wide
range of crops and farming terms; the dataset was read first, and the 52% that
suited the purpose was kept, 74,697 of 143,875 rows. Stage 2 trains on the
verified corpus for correctness, safety and African context. Every weight in the
model changed, on a single rented 48 GB GPU in about five hours.

**The model was measured on modest hardware as well as the reference machine.** In the ADTC profiler's
own Docker image on 4 vCPUs it writes **15.7 tokens a second in 1.07 GB of
memory**. On a laptop with a 2016 Intel i5, older than the reference spec, it
writes **about 12 tokens a second**, twice reading speed, with no throttling.

## Challenges

**Verified African farming data is scarce, so the corpus had to be written.** The
open agronomy datasets available either described farms on other continents or
declared no licence; the closest match, an East Africa agronomy set, had no
licence, so it was left out. Bulk generation gave quantity without quality: one
Swahili batch of 2,500 rows held only 121 distinct answers, and a later batch of
22,300 rows turned out to be about 700 answers with place names swapped. The
6,703 rows that shipped were written and fact-checked one file at a time.

**The public data that does exist needed heavy cleaning.** The largest usable
agricultural dataset, AI71ai/agrillm-train-146k, was about half suitable: 40,043
rows had no agricultural content, 21,815 carried leftover prompt scaffolding, and
2,571 stated pesticide or fertiliser rates as fact. The 74,697 rows that suited
the purpose were kept and used for breadth.

**Safe dosing advice differs from country to country.** Pesticide registrations
and rates vary by country, and a wrong rate damages a crop or poisons the person
spraying. Much of the available data stated doses anyway, so hand-written
examples taught the model to send dose questions to the product label and the
local agrodealer.

**It had to work even on modest hardware.** Advice matters most where resources
are scarce, so the model was built to run on an 8 GB laptop with integrated
graphics, and on older machines too. It had to download quickly and answer while
the farmer is still asking, and it does: about 12 tokens a second on a 2016 Intel
i5.

**African languages have even less verified data.** Swahili training pairs were
generated and verified, and there were too few for a 1.5B model to learn grammar
from; it produced repeating phrases. AgriLLM answers in English today, and
serving farmers in their own languages needs a much larger verified corpus.

## What was learned

Local models can reach communities that cloud AI leaves out. A 940 MB model on a
second-hand laptop gave useful, safe farming advice with the network switched
off, even on modest hardware. The gap
between farmers and good advice can be narrowed with the resources at hand: a
budget machine, open-source tools, and knowledge written carefully for the
people who will use it.

The same approach reaches beyond agriculture. Wherever expert help is scarce and
connectivity is costly, as in health, education or small business, a small local
model built around real needs can put a first answer within reach. AgriLLM showed
that this is practical today.

## What's next for AgriLLM

The next steps aim at the same farmers, starting with the one safety error left.
The shipped model advises keeping pesticide-contaminated clothing on after a
spill; it must come off, and that is the first corpus fix. After that: stronger
diagnosis, currently the weakest category at 59.4%; Swahili with enough data to
teach grammar as well as vocabulary; fact-checking against KALRO and FAO
material; and offline access to a farmer's own records, so the advice knows what
was planted in that field last season.

The first field test will be on the farm that inspired it.

---

**Full technical report, benchmarks and reproducibility instructions:**
[REPORT.md](https://github.com/shem2019/adtc-2026-agrillm/blob/main/REPORT.md)
