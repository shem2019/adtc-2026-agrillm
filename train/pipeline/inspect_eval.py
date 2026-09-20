#!/usr/bin/env python3
"""Show what the model ACTUALLY said on the prompts it failed.

    python3 train/pipeline/inspect_eval.py <eval_result.json> [--category safety_critical]
    python3 train/pipeline/inspect_eval.py train/pipeline/runs/fullft-lr1e-5/eval/final.json

A score of 33% with five safety-critical failures has at least three completely
different explanations, and they need opposite responses:

  1. the model genuinely gives dangerous answers  -> the training did not take
  2. the model gives good answers the rules miss  -> the harness is miscalibrated
  3. the model emits degenerate text              -> something broke in training
                                                     or quantisation

You cannot tell which from a percentage. You can tell instantly from reading
four generations. This prints them, next to the specific rule each one tripped.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("result", type=Path)
    ap.add_argument("--category", default=None, help="only this category")
    ap.add_argument("--max-chars", type=int, default=700)
    ap.add_argument("--max-items", type=int, default=8)
    args = ap.parse_args()

    if not args.result.exists():
        print(f"not found: {args.result}", file=sys.stderr)
        return 1

    d = json.loads(args.result.read_text(encoding="utf-8"))
    results = d["results"]

    print(f"file     : {args.result}")
    print(f"profile  : {d.get('profile')}  sampling={d.get('sampling')}  samples={d.get('samples')}")
    print(f"overall  : {d.get('overall', 0):.1%}")

    print("\nby category")
    for cat, score in sorted(d.get("by_category", {}).items()):
        n = sum(1 for r in results if r["category"] == cat)
        fails = sum(1 for r in results if r["category"] == cat and r["score"] < 1)
        print(f"  {cat:20} {score:6.1%}   {fails}/{n} prompts imperfect")

    # Which rules are doing the failing? If one rule accounts for most of it,
    # suspect the rule before suspecting the model.
    rule_counts: Counter = Counter()
    for r in results:
        for s in r["samples"]:
            for f in s["fails"]:
                rule_counts[f.split(":")[0].strip()[:60]] += 1
    if rule_counts:
        print("\nmost frequent failure reasons across all samples")
        for reason, n in rule_counts.most_common(10):
            print(f"  {n:4d}x  {reason}")

    failed = [r for r in results if r["score"] < 1]
    if args.category:
        failed = [r for r in failed if r["category"] == args.category]
    failed.sort(key=lambda r: (r["category"] != "safety_critical", r["score"]))

    print(f"\n{'=' * 74}\nFAILING PROMPTS ({len(failed)}) - showing up to {args.max_items}\n{'=' * 74}")

    for r in failed[: args.max_items]:
        print(f"\n[{r['category']}] {r['id']}   score {r['score']:.2f} "
              f"({r['passed']}/{r['n']} samples clean)")
        print(f"  PROMPT: {r['prompt'][:200]}")
        if r.get("followup"):
            print(f"  FOLLOWUP: {r['followup'][:160]}")

        bad = next((s for s in r["samples"] if s["fails"]), None)
        if bad:
            print(f"  TRIPPED: {'; '.join(bad['fails'][:4])}")
            text = bad["text"].strip().replace("\n", "\n            ")
            print(f"  MODEL SAID:\n            {text[:args.max_chars]}")
            if len(bad["text"]) > args.max_chars:
                print(f"            ... [{len(bad['text'])} chars total]")

        good = next((s for s in r["samples"] if not s["fails"]), None)
        if good:
            print(f"  (a clean sample also occurred - this prompt is INCONSISTENT, "
                  f"{r['passed']}/{r['n']})")

    print(f"\n{'=' * 74}")
    print("Read four of these before changing anything. If the answers are sensible")
    print("and the rules are pedantic, fix the rules. If the answers are dangerous")
    print("or degenerate, the model is the problem.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
