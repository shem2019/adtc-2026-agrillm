#!/usr/bin/env python3
"""Reshape the merged corpus so it teaches the register the judges actually read.

    python3 train/rebalance.py --dry-run
    python3 train/rebalance.py

The first fine-tune collapsed: judged answers fell from ~220 words to ~20, and
the "cannot be determined" template leaked into questions that were perfectly
answerable. Nothing was wrong with the training run — LoRA fit the corpus it was
given, and that corpus has a median answer of 51 words, 39% of answers under 40
words, and 13.4% refusals. Length and refusal rate are learned behaviours, so
training longer on the same mix drives the model further into that failure, not
out of it.

This rewrites train.jsonl (valid/test are left alone, so evaluation stays
comparable across runs) with three caps:

  * refusals capped as a fraction of the corpus, keeping the *ability* to say
    "cannot be determined" without making it a reflex,
  * answers below a word floor dropped, so the model stops learning to stop,
  * the "Therefore:" template capped, since it dominated 59% of examples and
    surfaced verbatim in prose.

The originals are copied to *.orig.jsonl on first run, so this is reversible.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

REPO = Path(__file__).parent.parent
DATA = REPO / "train" / "data"

REFUSAL_MARK = "cannot be determined"
TEMPLATE_MARK = "Therefore:"


def answer_of(text: str) -> str:
    return text.split("Answer:", 1)[1] if "Answer:" in text else text


def words(text: str) -> int:
    return len(answer_of(text).split())


def describe(rows: list[dict], label: str) -> None:
    n = len(rows)
    if not n:
        print(f"  {label}: empty")
        return
    lens = sorted(words(r["text"]) for r in rows)
    ref = sum(1 for r in rows if REFUSAL_MARK in r["text"])
    tpl = sum(1 for r in rows if TEMPLATE_MARK in r["text"])
    short = sum(1 for x in lens if x < 40)
    print(
        f"  {label:9} n={n:5d}  median={lens[n // 2]:3d}w  p90={lens[int(.9 * n)]:3d}w  "
        f"<40w={100 * short / n:4.1f}%  refusals={100 * ref / n:4.1f}%  template={100 * tpl / n:4.1f}%"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-words", type=int, default=45,
                    help="Drop answers shorter than this. The judged register is 150-250 words.")
    ap.add_argument("--max-refusal-frac", type=float, default=0.03,
                    help="Cap 'cannot be determined' as a fraction of the corpus.")
    ap.add_argument("--max-template-frac", type=float, default=0.35,
                    help="Cap the 'Therefore:' template as a fraction of the corpus.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry-run", action="store_true",
                    help="Report what would change without writing anything.")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    src = DATA / "train.jsonl"
    orig = DATA / "train.orig.jsonl"

    # Always rebalance from the pristine corpus, so re-running with different
    # caps never compounds an earlier filter.
    read_from = orig if orig.exists() else src
    rows = [json.loads(line) for line in read_from.read_text().splitlines() if line.strip()]

    print(f"reading {read_from.name}")
    describe(rows, "before")

    # 1. Length floor. This is the single change that most directly undoes the
    #    "answers in 19 words" failure.
    kept = [r for r in rows if words(r["text"]) >= args.min_words]
    dropped_short = len(rows) - len(kept)

    # 2. Refusal cap. Keep a sample rather than removing the class entirely —
    #    refusing when an input genuinely is missing is correct behaviour, it
    #    just must not be the model's default move.
    refusals = [r for r in kept if REFUSAL_MARK in r["text"]]
    others = [r for r in kept if REFUSAL_MARK not in r["text"]]
    # Budget is a fraction of the final corpus, which is (others + budget).
    budget = int(len(others) * args.max_refusal_frac / max(1e-9, 1 - args.max_refusal_frac))
    rng.shuffle(refusals)
    dropped_refusals = max(0, len(refusals) - budget)
    kept = others + refusals[:budget]

    # 3. Template cap, applied last so it measures the real final mix.
    tpl = [r for r in kept if TEMPLATE_MARK in r["text"]]
    non_tpl = [r for r in kept if TEMPLATE_MARK not in r["text"]]
    tpl_budget = int(len(non_tpl) * args.max_template_frac / max(1e-9, 1 - args.max_template_frac))
    rng.shuffle(tpl)
    dropped_template = max(0, len(tpl) - tpl_budget)
    kept = non_tpl + tpl[:tpl_budget]

    rng.shuffle(kept)

    print(f"\n  dropped {dropped_short} short (<{args.min_words}w), "
          f"{dropped_refusals} excess refusals, {dropped_template} excess template")
    describe(kept, "after")

    if not kept:
        print("\nrefusing to write an empty corpus — loosen the caps")
        return 1

    if args.dry_run:
        print("\n(dry run — nothing written)")
        return 0

    if not orig.exists():
        shutil.copy2(src, orig)
        print(f"\n  original preserved at {orig.name}")

    with src.open("w") as fh:
        for r in kept:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  wrote {len(kept)} examples to {src.name}")
    print("\nvalid.jsonl and test.jsonl untouched, so val loss stays comparable across runs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
