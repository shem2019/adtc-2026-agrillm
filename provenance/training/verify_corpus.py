#!/usr/bin/env python3
"""Gate-check a corpus file the way it should have been checked the first time.

    python3 train/verify_corpus.py train/african/_clean/safety-poisoning.jsonl
    python3 train/verify_corpus.py train/african/_clean/*.jsonl

Why this exists: an earlier pass reported "100% distinct answers" on a
templated corpus where distinctness was manufactured with a "Ref {i}:" serial
number prepended to two or three underlying strings. Raw-string distinctness
passed trivially and meant nothing. This checks the SHAPE of the answer after
stripping exactly the kind of cosmetic wrapper that gamed the last check, so
it cannot be re-gamed the same way. It also enforces the safety and register
rules from corpus_refinements.md that are cheap to check mechanically.

This is not a substitute for reading rows yourself. It is a floor.
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

# ---- shape normalisation -----------------------------------------------
# Strip exactly the family of wrapper that defeated the last check: a leading
# "Ref 1327:", "Advisory #618 [Pan-African Block 18]:", "Response 4:" etc.
ID_PREFIX = re.compile(
    r'^\s*(ref|advisory|response|answer|guidance|note|entry)\.?\s*#?\s*\d+'
    r'\s*(\[[^\]]*\])?\s*:?\s*', re.I)
BLOCK_TAG = re.compile(r'\[[^\]]*block\s*\d+[^\]]*\]|\bzone\s*\d+\b|\bdistrict\s*\d+\b', re.I)

# A second, milder templating pattern found in the RESTORED (non-fraudulent)
# original corpus: the same underlying fact repeated with only a
# "in <Town>, <Country>" location and/or a "for <flock/crop type> in ..."
# role clause swapped out. corpus_refinements.md #2 is explicit that place
# names carry zero content value, so this pattern must not count toward
# distinctness either, even though it's a much milder version of the
# fraud-corpus problem (paraphrase templating, not literal serial numbers).
LOCATION_INLINE = re.compile(
    r'\b(?:in|across|for|throughout|near|within|at|around)\s+'
    r'[A-Z][\w\'-]+(?:\s+[A-Z][\w\'-]+){0,2},\s*[A-Z][\w\'-]+(?:\s+[A-Z][\w\'-]+)?\b')
LEAD_ROLE_CLAUSE = re.compile(
    r'^(for|in)\s+[a-z0-9][^.,]{2,80}?,\s*(regarding[^.,]{2,60},\s*)?', re.I)

def shape(answer: str) -> str:
    a = ID_PREFIX.sub('', answer)
    a = BLOCK_TAG.sub('', a)
    a = LEAD_ROLE_CLAUSE.sub('', a)
    a = LOCATION_INLINE.sub('', a)
    a = re.sub(r'\d+', 'N', a)          # numbers vary legitimately; ignore them
    a = re.sub(r'\s+', ' ', a).strip().lower()
    return a

def shape_prefix(answer: str, n_chars: int = 180) -> str:
    return shape(answer)[:n_chars]

# ---- banned abstractions (corpus_refinements.md §9) ---------------------
BANNED_ABSTRACT = re.compile(
    r'\bproper ppe\b|\bappropriate protective equipment\b|\bprotective gear\b|'
    r'\bsuitable clothing\b|\bsafety precautions\b|\bhandle with care\b|'
    r'\bfollow safety guidelines\b|\buse caution\b|\bstore properly\b|'
    r'\bstore appropriately\b|\bdispose of appropriately\b|'
    r'\btake necessary precautions\b|\bobserve safety measures\b', re.I)

# ---- dose / rate regex (never allowed for CHEMICAL/VET treatment rates) --
# Deliberately narrower than a bare "number + unit" match: §10 of
# corpus_refinements.md requires the model to state fertiliser and seed
# rates CONFIDENTLY, and a plant's own nitrogen fixation (a fact, not an
# instruction) also legitimately carries a kg/ha figure. Both must NOT be
# flagged. What must be flagged is a rate for something you SPRAY, INJECT,
# MIX or ADMINISTER as a pesticide, acaricide, dewormer or drug.
DOSE_VERB = r'(apply|spray|mix|inject|injection|administer|dose|soluble powder)'
CHEM_NOUN = (r'(fungicide|insecticide|pesticide|acaricide|herbicide|'
              r'dewormer|antibiotic|drug|chemical|active ingredient|'
              r'mg per kg|mg/kg|ml per \d|% (wp|ec|sc)\b)')
FERTILISER_TERMS = re.compile(
    r'\b(CAN|DAP|NPK|TSP|urea|manure|compost|fertiliser|fertilizer|'
    r'top-?dress|nitrogen(?!ous herbicide))\b', re.I)
_DOSE_RATE_RAW = re.compile(
    rf'(?:{DOSE_VERB}[^.]{{0,80}}\d+(\.\d+)?\s*(ppm|ml/l|ml per|l/ha|kg/ha|g/l)'
    rf'|\d+(\.\d+)?\s*(mg per kg|mg/kg|ppm)'
    rf'|{CHEM_NOUN}[^.]{{0,60}}\d+(\.\d+)?\s*(ppm|ml/l|ml per|kg/ha|g/l))',
    re.I)

class _DoseRate:
    """Wraps the raw dose regex to exclude fertiliser context. Fertiliser and
    seed rates must be stated CONFIDENTLY per corpus_refinements.md §10 --
    only pesticide/acaricide/veterinary rates are banned."""
    def search(self, text):
        m = _DOSE_RATE_RAW.search(text)
        if not m:
            return None
        window = text[max(0, m.start() - 40):m.end() + 10]
        if FERTILISER_TERMS.search(window) and not re.search(
                r'\b(fungicide|insecticide|pesticide|acaricide|mg per kg|mg/kg)\b',
                window, re.I):
            return None  # fertiliser rate, not a chemical/vet dose -- allowed
        return m

DOSE_RATE = _DoseRate()

# ---- generation-tag leakage (the specific defect found this round) ------
GEN_TAG = re.compile(
    r'\bref\s*\d+\b|\badvisory\s*\d+\b|pan-?african block|\[emergency medical\]|'
    r'\[dosage refusal\]|\[safety handling\]|\[veterinary safety\]|'
    r'\[uncertainty advisory\]|\[concrete safety advisory\]', re.I)

# ---- markdown ----
MARKDOWN = re.compile(r'(\*\*|^#{1,6}\s|^\s*[-*]\s\*\*)', re.MULTILINE)

def unique_token_ratio(text: str) -> float:
    toks = re.findall(r'\w+', text.lower())
    return len(set(toks)) / len(toks) if len(toks) >= 20 else 1.0

def near_dup_ratio(a: str, b: str) -> float:
    return SequenceMatcher(None, a[:200].lower(), b[:200].lower()).ratio()


def load(path: Path) -> list[dict]:
    rows = []
    for lineno, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as e:
            print(f"  ✗ {path.name}:{lineno} malformed JSON: {e}")
            continue
        if 'messages' in row:
            q = row['messages'][0]['content']
            a = row['messages'][-1]['content']
        else:
            q, a = row.get('question', ''), row.get('answer', '')
        rows.append({'q': q, 'a': a, 'lineno': lineno})
    return rows


def verify(path: Path) -> bool:
    rows = load(path)
    n = len(rows)
    if n == 0:
        print(f"{path.name}: EMPTY")
        return False

    shapes = Counter(shape(r['a']) for r in rows)
    distinct_shapes = len(shapes)
    distinct_pct = distinct_shapes / n
    max_reuse = shapes.most_common(1)[0][1]
    max_reuse_pct = max_reuse / n
    # Small files: a single occurrence is >2% trivially. Use an absolute floor
    # so a 16-row batch isn't failed for having zero actual duplicates.
    reuse_floor = max(2, round(0.02 * n))

    gen_tag_hits = sum(1 for r in rows if GEN_TAG.search(r['a']) or GEN_TAG.search(r['q']))
    banned_hits = [(r['lineno'], m.group(0)) for r in rows
                   for m in [BANNED_ABSTRACT.search(r['a'])] if m]
    dose_hits = [(r['lineno'], m.group(0)) for r in rows
                 for m in [DOSE_RATE.search(r['a'])] if m]
    md_hits = sum(1 for r in rows if MARKDOWN.search(r['a']))

    # shape-prefix near-duplicate scan (cheap O(n log n)-ish via bucket on prefix)
    prefixes = Counter(shape_prefix(r['a']) for r in rows)
    top_prefix_reuse = prefixes.most_common(1)[0][1] if prefixes else 0

    ok = True
    print(f"\n{'='*70}\n{path.name}  ({n} rows)")
    print(f"  distinct answer SHAPES      : {distinct_shapes}/{n} = {distinct_pct:.1%}"
          f"   {'✓' if distinct_pct >= 0.60 else '✗ FAIL (need ≥60%)'}")
    ok &= distinct_pct >= 0.60
    print(f"  worst shape reused          : {max_reuse}x ({max_reuse_pct:.1%} of file)"
          f"   {'✓' if max_reuse <= reuse_floor else f'✗ FAIL (need ≤{reuse_floor})'}")
    ok &= max_reuse <= reuse_floor
    print(f"  worst 180-char prefix reused : {top_prefix_reuse}x"
          f"   {'✓' if top_prefix_reuse <= reuse_floor else f'✗ FAIL (need ≤{reuse_floor})'}")
    ok &= top_prefix_reuse <= reuse_floor
    print(f"  generation-tag leakage       : {gen_tag_hits} rows"
          f"   {'✓' if gen_tag_hits == 0 else '✗ FAIL — templating artefact'}")
    ok &= gen_tag_hits == 0
    print(f"  banned abstractions          : {len(banned_hits)} rows"
          f"   {'✓' if not banned_hits else '✗ FAIL'}")
    ok &= not banned_hits
    for ln, m in banned_hits[:5]:
        print(f"      line {ln}: {m!r}")
    print(f"  numeric chemical dose stated  : {len(dose_hits)} rows"
          f"   {'✓' if not dose_hits else '✗ FAIL — never allowed'}")
    ok &= not dose_hits
    for ln, m in dose_hits[:5]:
        print(f"      line {ln}: {m!r}")
    print(f"  markdown in answers           : {md_hits} rows"
          f"   {'✓' if md_hits == 0 else '✗ FAIL'}")
    ok &= md_hits == 0

    print(f"  → {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    paths = [Path(p) for p in sys.argv[1:]]
    if not paths:
        print("usage: verify_corpus.py <file.jsonl> [...]")
        return 1
    results = {p: verify(p) for p in paths}
    total = sum(len(load(p)) for p in paths)
    passed = sum(1 for ok in results.values() if ok)
    print(f"\n{'='*70}")
    print(f"{passed}/{len(paths)} files pass, {total} rows total")
    return 0 if all(results.values()) else 1


if __name__ == '__main__':
    sys.exit(main())
