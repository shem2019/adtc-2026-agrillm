#!/usr/bin/env python3
"""Validate and clean LLM-generated JSONL before it reaches training.

    python3 train/validate_african.py
    python3 train/validate_african.py --strict     # exit 1 if anything rejected

Reads  train/african/*.jsonl
Writes train/african/_clean/*.jsonl   <- prepare_data.py reads this

Generated data fails in patterns, not at random: a whole batch wrapped in a code
fence, every answer opening with "Certainly", the same 40 questions rephrased.
This reports failures BY CATEGORY WITH COUNTS so you can paste the specific
complaint back to the generator and fix the source, rather than silently
discarding a third of a batch and never knowing why.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, deque
from difflib import SequenceMatcher
from pathlib import Path

HERE = Path(__file__).parent
SRC = HERE / "african"
OUT = SRC / "_clean"

# An answer may legitimately recur once or twice across 13,000 rows (short
# factual answers collide). Twenty times means a template, not a corpus:
# swahili.jsonl shipped 2,500 rows carrying only 121 distinct answers, each
# recycled ~20x. Deduping on QUESTIONS alone missed it entirely, because the
# generator varied the question and reused the answer.
# Default is now effectively OFF. It was added to catch a 20.7x templated batch,
# but it also cut the English corpus from 13,052 to 7,260 rows -- and answers
# that legitimately recur are how a model consolidates a fact, not noise. The
# diversity ratio is still reported so templating stays visible.
MAX_ANSWER_REUSE = 999

MIN_Q, MAX_Q = 20, 400
MIN_A, MAX_A = 15, 2800

FILLER_OPEN = re.compile(
    r"^\s*(certainly|sure|of course|absolutely|great question|"
    r"here'?s?( is)?( a| an| the)?\s*(brief |short |detailed |helpful )?"
    r"(overview|answer|explanation|guide)|i'?d be happy to)",
    re.IGNORECASE,
)
BANNED = re.compile(
    r"(as an ai|language model|i cannot provide|\\boxed|\$\$|"
    r"^\s*```|gemini|openai|chatgpt)",
    re.IGNORECASE,
)
MARKDOWN = re.compile(r"(\*\*|^#{1,6}\s)", re.MULTILINE)
AGRI = re.compile(
    r"\b(crop|farm|soil|seed|plant|harvest|fertili|irrigat|pest|weed|maize|"
    r"cassava|bean|rice|sorghum|millet|coffee|tea|banana|potato|tomato|"
    r"livestock|cattle|goat|sheep|poultry|chicken|manure|compost|agronom|"
    r"agricultur|yield|germinat|nitrogen|phosphor|potassium|acre|hectare|"
    r"mulch|rotation|pesticide|herbicide|fungicide|drought|rain|variet|"
    r"disease|larva|aphid|weevil|blight|wilt|spacing|planting|grazing|"
    r"extension|fodder|vaccin|dewor|tick|striga|armyworm|storage|aflatoxin|"
    # Swahili agricultural vocabulary -- without these the validator rejects
    # every Swahili row as having no agricultural content.
    r"kilimo|mkulima|shamba|udongo|mbegu|mbolea|samadi|wadudu|ugonjwa|kupanda|"
    r"mavuno|kuvuna|umwagiliaji|mahindi|maharagwe|mihogo|viazi|mtama|ndizi|"
    r"kahawa|nyanya|vitunguu|mbuzi|kuku|mifugo|viwavijeshi|mvua|msimu|mimea|"
    r"mazao|ugani)",
    re.IGNORECASE,
)
# East African signal — the whole reason this corpus exists.
EAST_AFRICA = re.compile(
    r"\b(kenya|tanzania|uganda|rwanda|ethiopia|burundi|somalia|south sudan|"
    r"nakuru|kitale|eldoret|kisumu|meru|embu|machakos|kakamega|bungoma|"
    r"arusha|mbeya|morogoro|iringa|dodoma|mwanza|kilimanjaro|"
    r"kampala|mbale|gulu|masaka|jinja|kigali|musanze|oromia|amhara|"
    r"long rains|short rains|masika|vuli|belg|meher|shilling|"
    r"east africa|kalro|naads|tari|rab|icipe|cimmyt|iita|"
    r"fall armyworm|striga|cassava mosaic|brown streak|east coast fever|"
    r"bacterial wilt|newcastle)",
    re.IGNORECASE,
)


def similar(a: str, b: str) -> float:
    la, lb = len(a), len(b)
    if la == 0 or lb == 0 or (2 * min(la, lb) / (la + lb)) < 0.85:
        return 0.0
    sm = SequenceMatcher(None, a[:160].lower(), b[:160].lower())
    if sm.quick_ratio() < 0.85:
        return 0.0
    return sm.ratio()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true")
    ap.add_argument("--dup-threshold", type=float, default=0.90)
    ap.add_argument("--max-answer-reuse", type=int, default=MAX_ANSWER_REUSE,
                    help="Reject an answer after it appears this many times. "
                         "Default is effectively off; set to 2 to hunt templates.")
    args = ap.parse_args()

    if not SRC.exists():
        print(f"no {SRC}/ — create it and save Gemini's batches there as .jsonl")
        return 1

    files = sorted(p for p in SRC.glob("*.jsonl") if p.parent.name == "african")
    if not files:
        print(f"no .jsonl files in {SRC}/")
        return 1

    OUT.mkdir(exist_ok=True)
    reasons: Counter[str] = Counter()
    examples: dict[str, str] = {}
    kept_all: list[tuple[str, str]] = []
    seen_q: set[str] = set()
    answer_uses: Counter[str] = Counter()
    recent_q: deque[str] = deque(maxlen=400)
    lengths = Counter()
    ea_hits = 0

    def reject(why: str, sample: str) -> None:
        reasons[why] += 1
        examples.setdefault(why, sample[:110])

    for path in files:
        kept = []
        raw = path.read_text(encoding="utf-8")
        # A whole batch wrapped in a code fence is the single most common
        # generator failure. Strip it rather than rejecting 250 good entries.
        raw = re.sub(r"^\s*```(?:json|jsonl)?\s*$", "", raw, flags=re.MULTILINE)

        for lineno, line in enumerate(raw.splitlines(), 1):
            line = line.strip().rstrip(",")
            if not line or line in ("[", "]"):
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                reject("malformed JSON", f"{path.name}:{lineno} {e.msg}")
                continue
            if not isinstance(row, dict):
                reject("not an object", line)
                continue

            q = str(row.get("question", "")).strip()
            a = str(row.get("answer", "")).strip()

            if not q or not a:
                reject("missing question or answer", line); continue
            if not (MIN_Q <= len(q) <= MAX_Q):
                reject(f"question length outside {MIN_Q}-{MAX_Q}", q); continue
            if not (MIN_A <= len(a) <= MAX_A):
                reject(f"answer length outside {MIN_A}-{MAX_A}", a); continue
            if FILLER_OPEN.search(a):
                reject("filler opener", a); continue
            if BANNED.search(a) or BANNED.search(q):
                reject("banned phrase / code fence / LaTeX", a); continue
            if not AGRI.search(q + " " + a):
                reject("no agricultural content", q); continue

            key = re.sub(r"[^a-z0-9 ]", "", q.lower()).strip()
            if key in seen_q:
                reject("exact duplicate question", q); continue

            near = next((pq for pq in recent_q
                         if similar(pq, key) > args.dup_threshold), None)
            if near:
                reject("near-duplicate question", q); continue

            akey = re.sub(r"[^a-z0-9 ]", "", a.lower()).strip()[:200]
            answer_uses[akey] += 1
            if answer_uses[akey] > args.max_answer_reuse:
                reject(f"answer reused >{args.max_answer_reuse}x (templated)", a); continue

            seen_q.add(key)
            recent_q.append(key)
            if MARKDOWN.search(a):
                a = MARKDOWN.sub("", a).strip()      # salvage, don't discard

            kept.append({"question": q, "answer": a})
            kept_all.append((q, a))
            lengths["long" if len(a) > 900 else "medium" if len(a) > 300 else "short"] += 1
            if EAST_AFRICA.search(q + " " + a):
                ea_hits += 1

        if kept:
            dest = OUT / path.name
            dest.write_text(
                "\n".join(json.dumps(r, ensure_ascii=False) for r in kept) + "\n",
                encoding="utf-8")
        print(f"  {path.name:<34} kept {len(kept):>5}")

    total_in = sum(reasons.values()) + len(kept_all)
    print(f"\n{'=' * 62}")
    print(f"  accepted {len(kept_all):,} / {total_in:,} "
          f"({len(kept_all) / max(total_in, 1):.0%})")
    print(f"  written to {OUT.relative_to(HERE.parent)}/")

    if reasons:
        print("\n  rejected:")
        for why, n in reasons.most_common():
            print(f"    {n:>5}  {why}")
            print(f"           e.g. {examples[why]}")

    if kept_all:
        uniq_a = len({a for _, a in kept_all})
        ratio = len(kept_all) / uniq_a
        print(f"\n  answer diversity: {uniq_a:,} unique answers for "
              f"{len(kept_all):,} rows ({ratio:.1f}x reuse)")
        if ratio > 1.8:
            print("    ^ HIGH. The generator is likely filling a template rather "
                  "than\n      writing distinct answers. Inspect before training.")

    print("\n  length mix (target roughly 30 long / 50 medium / 20 short):")
    for k in ("long", "medium", "short"):
        n = lengths[k]
        print(f"    {k:<7}{n:>6}  {n / max(len(kept_all), 1):>5.0%}")

    ea_pct = ea_hits / max(len(kept_all), 1)
    print(f"\n  East African specificity: {ea_hits:,} / {len(kept_all):,} "
          f"({ea_pct:.0%})")
    if ea_pct < 0.5:
        print("    ^ LOW. This corpus exists to add African context. If most "
              "entries\n      could have been written about anywhere, ask the "
              "generator to name\n      regions, seasons and locally-relevant "
              "pests explicitly.")

    if args.strict and reasons:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
