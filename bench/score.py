#!/usr/bin/env python3
"""Rank sweep results using the ADTC leaderboard formula.

Formula, verbatim from adtc_profiler's README and source:

    S_total = 0.50*S_acc + 0.30*S_perf + 0.20*S_eff - P_thermal
    S_perf  = min(TPS / 15.0, 1.0) * 100          <- CAPPED, not ranked
    S_eff   = max(0, (7.0 - peak_rss_gb) / 7.0) * 100
    P_therm = 10 if throttled or core temp > 85C

The cap on S_perf is the whole game. Every token/sec above 15 is worth exactly
zero points, so the optimum is the largest, most accurate model that still
clears 15 tok/s on the audit VM.

Usage:
    python3 score.py                  # rank results/, assume S_acc unknown
    python3 score.py --acc 72         # assume a uniform accuracy for comparison
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

TPS_REFERENCE = 15.0
RAM_LIMIT_GB = 7.0

# Your dev box is almost certainly faster than a 10th-gen i5 with DDR4-3200.
# The comparator FAILS a submission when participant and audit throughput differ
# by more than 50%, so a locally-measured 45 tok/s against an audit-measured
# 16 tok/s is a disqualifying -64%. Treat this as the de-rating you expect
# between your machine and theirs, and prefer candidates that clear 15 tok/s
# even after it is applied.
DEFAULT_DERATE = 2.0


def s_perf(tps: float) -> float:
    return min(tps / TPS_REFERENCE, 1.0) * 100.0


def s_eff(peak_rss_gb: float) -> float:
    return max(0.0, (RAM_LIMIT_GB - peak_rss_gb) / RAM_LIMIT_GB) * 100.0


def load(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as e:
        print(f"  ! skipping {path.name}: {e}", file=sys.stderr)
        return None


def extract(report: dict) -> dict:
    thr = report.get("throughput") or {}
    mem = report.get("memory") or {}
    therm = report.get("cpu_thermal") or {}
    info = report.get("model_info") or {}
    acc_rows = report.get("accuracy") or []

    peak_mb = mem.get("peak_rss_mb") or 0.0
    # lm-eval reports accuracy in 0..1; the leaderboard wants 0..100.
    acc = None
    if acc_rows:
        raw = acc_rows[0].get("score")
        if isinstance(raw, (int, float)):
            acc = raw * 100.0 if raw <= 1.0 else float(raw)

    throttled = bool(therm.get("throttled")) or (
        isinstance(therm.get("max_temp_c"), (int, float)) and therm["max_temp_c"] > 85
    )

    return {
        "tps": float(thr.get("tokens_per_second_generation") or 0.0),
        "ttft_ms": float(thr.get("first_token_latency_ms") or 0.0),
        "peak_gb": peak_mb / 1024.0,
        "acc": acc,
        "throttled": throttled,
        "params": info.get("params_count"),
        "params_match": info.get("params_match"),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(Path(__file__).parent / "results"))
    ap.add_argument(
        "--acc",
        type=float,
        default=None,
        help="Assume this S_acc (0-100) for candidates with no accuracy row, "
        "so throughput/memory can be compared on equal footing.",
    )
    ap.add_argument(
        "--derate",
        type=float,
        default=DEFAULT_DERATE,
        help=f"Divide measured TPS by this to estimate audit-VM throughput "
        f"(default {DEFAULT_DERATE}).",
    )
    args = ap.parse_args()

    results_dir = Path(args.results)
    files = sorted(results_dir.glob("*.json"))
    if not files:
        print(f"no result files in {results_dir} — run ./sweep.sh first")
        return 1

    rows = []
    for path in files:
        report = load(path)
        if report is None:
            continue
        m = extract(report)
        acc = m["acc"] if m["acc"] is not None else args.acc
        derated_tps = m["tps"] / args.derate if args.derate > 0 else m["tps"]

        sp_raw = s_perf(m["tps"])
        sp_der = s_perf(derated_tps)
        se = s_eff(m["peak_gb"])
        pen = 10.0 if m["throttled"] else 0.0

        total = None
        if acc is not None:
            total = 0.50 * acc + 0.30 * sp_der + 0.20 * se - pen

        rows.append(
            {
                "label": path.stem,
                "tps": m["tps"],
                "tps_der": derated_tps,
                "peak_gb": m["peak_gb"],
                "acc": acc,
                "sp_raw": sp_raw,
                "sp_der": sp_der,
                "se": se,
                "pen": pen,
                "total": total,
                "headroom": m["tps"] / TPS_REFERENCE,
                "params_match": m["params_match"],
            }
        )

    rows.sort(key=lambda r: (r["total"] is not None, r["total"] or 0), reverse=True)

    print()
    print(f"  de-rating measured TPS by {args.derate}x to estimate the audit VM")
    print(f"  S_perf saturates at {TPS_REFERENCE} tok/s — more is worth nothing")
    print()
    hdr = (
        f"{'candidate':<22}{'TPS':>8}{'est.aud':>9}{'x15':>7}"
        f"{'RAM GB':>9}{'S_acc':>8}{'S_perf':>8}{'S_eff':>8}{'TOTAL':>9}"
    )
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for r in rows:
        acc_s = f"{r['acc']:.1f}" if r["acc"] is not None else "  ?"
        tot_s = f"{r['total']:.1f}" if r["total"] is not None else "    ?"
        flag = ""
        if r["tps_der"] < TPS_REFERENCE:
            flag = "  ← under 15 after de-rating"
        if r["pen"]:
            flag += "  ⚠ THERMAL -10"
        if r["params_match"] is False:
            flag += "  ⚠ params mismatch"
        print(
            f"{r['label']:<22}{r['tps']:>8.1f}{r['tps_der']:>9.1f}"
            f"{r['headroom']:>7.1f}{r['peak_gb']:>9.2f}{acc_s:>8}"
            f"{r['sp_der']:>8.1f}{r['se']:>8.1f}{tot_s:>9}{flag}"
        )

    print()
    if any(r["acc"] is None for r in rows):
        print(
            "  Accuracy is 50% of the score and is missing for some candidates.\n"
            "  Re-run with ./sweep.sh --accuracy, or pass --acc N to compare the\n"
            "  performance half on equal footing."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
