#!/usr/bin/env python3
"""Diagnose one source before tuning filters against it.

    python3 train/inspect_source.py manifesta/verified-agronomy-17k

Reports the actual length distribution and which filter is responsible for each
dropped row, so the cap is set from evidence rather than guessed at twice.
"""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("prep", Path(__file__).parent / "prepare_data.py")
prep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prep)


def pct(values: list[int], p: float) -> int:
    if not values:
        return 0
    s = sorted(values)
    return s[min(int(len(s) * p), len(s) - 1)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("hf_id")
    ap.add_argument("--show", type=int, default=2, help="Print N sample rows.")
    args = ap.parse_args()

    from datasets import load_dataset
    ds = load_dataset(args.hf_id, split="train")
    print(f"\n{args.hf_id}: {len(ds):,} rows")
    print(f"columns: {list(ds.features)}\n")

    qs, as_, reasons = [], [], {}
    kept = 0
    over_by = []          # how far over MAX_A the long answers actually are
    for row in ds:
        q = prep.clean(prep._first_field(row, prep.Q_KEYS))
        a = prep.clean(prep._first_field(row, prep.A_KEYS))
        qs.append(len(q))
        as_.append(len(a))
        ok, why = prep.usable(q, a)
        if ok:
            kept += 1
        else:
            reasons[why] = reasons.get(why, 0) + 1
            if why == "length" and len(a) > prep.MAX_A:
                over_by.append(len(a))

    print(f"current filters: MIN_Q={prep.MIN_Q} MAX_Q={prep.MAX_Q} "
          f"MIN_A={prep.MIN_A} MAX_A={prep.MAX_A}")
    print(f"kept {kept:,} / {len(ds):,}\n")

    print("drop reasons:")
    for why, n in sorted(reasons.items(), key=lambda kv: -kv[1]):
        print(f"  {why:<24} {n:>7,}")

    print("\nanswer length distribution (characters):")
    for p in (0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99):
        print(f"  p{int(p*100):<3} {pct(as_, p):>7,}")
    print(f"  max  {max(as_) if as_ else 0:>7,}")

    print("\nquestion length distribution (characters):")
    for p in (0.50, 0.90, 0.99):
        print(f"  p{int(p*100):<3} {pct(qs, p):>7,}")

    if over_by:
        print(f"\n{len(over_by):,} answers exceed MAX_A={prep.MAX_A}. "
              f"Median of those: {pct(over_by, 0.5):,} chars")

    # What would each candidate cap recover? Length is only one gate, so this
    # re-runs the FULL filter chain at each setting rather than counting lengths.
    print("\nwhat different MAX_A values would keep (full filter chain):")
    original = prep.MAX_A
    for cap in (1200, 3000, 6000, 10000, 20000):
        prep.MAX_A = cap
        n = sum(1 for row in ds
                if prep.usable(prep.clean(prep._first_field(row, prep.Q_KEYS)),
                               prep.clean(prep._first_field(row, prep.A_KEYS)))[0])
        approx_tokens = cap // 4
        print(f"  MAX_A={cap:>6,} (~{approx_tokens:>5,} tok) -> {n:>7,} rows")
    prep.MAX_A = original

    if args.show:
        print("\n" + "=" * 70)
        for row in list(ds)[: args.show]:
            q = prep.clean(prep._first_field(row, prep.Q_KEYS))
            a = prep.clean(prep._first_field(row, prep.A_KEYS))
            print(f"\nQ ({len(q)} chars): {q[:200]}")
            print(f"A ({len(a)} chars): {a[:400]}...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
