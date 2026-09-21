#!/usr/bin/env python3
"""Draw the model journey: base model to shipped model, every run measured on
the same 24-prompt harness.

    python3 assets/make_journey_svg.py

Writes assets/journey.svg. Every number is read from provenance/logs/ so the
picture cannot drift from the evidence.
"""
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
EVAL = ROOT / "provenance" / "logs" / "eval"

# (file stem, short label, second label line, phase)
POINTS = [
    ("baseline__base-model",              "Qwen2.5-1.5B", "no training",  "base"),
    ("fullft-lr5e-6-2ep__checkpoint-408", "2 epochs",     "lr 5e-6",      "single"),
    ("fullft-lr5e-6__checkpoint-612",     "4 epochs",     "lr 5e-6",      "single"),
    ("fullft-lr1e-5__final",              "4 epochs",     "lr 1e-5",      "single"),
    ("fullft-lr2e-5__checkpoint-408",     "4 epochs",     "lr 2e-5",      "single"),
    ("fullft-2stage__checkpoint-1020",    "epoch 5",      "",             "two"),
    ("fullft-2stage__checkpoint-1224",    "epoch 6",      "loss 0.594",   "shipped"),
    ("fullft-2stage-x__final",            "+4 epochs",    "loss 0.021",   "rejected"),
]

# Phase bands drawn behind the points. (first index, last index, label, tint)
BANDS = [
    (0, 0, "Base model",                                    "#eaeef2"),
    (1, 4, "Trained on the 6,703 verified rows only",       "#f3eefc"),
    (5, 7, "Two-stage: 74,697 borrowed rows, then ours",    "#eaf2fb"),
]

PHASE = {
    "base":     ("#6e7781", "Base model"),
    "single":   ("#8250df", "Verified corpus only"),
    "two":      ("#0969da", "Two-stage"),
    "shipped":  ("#1a7f37", "Shipped"),
    "rejected": ("#cf222e", "Rejected — memorised"),
}

W, H = 1020, 590
L, R, T = 86, 30, 132         # margins
PLOT_B = 400                  # bottom of the score plot
SAFETY_Y = 452                # safety strip
FONT = ("-apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, "
        "sans-serif")


def load():
    out = []
    for stem, l1, l2, phase in POINTS:
        f = EVAL / f"{stem}.json"
        if not f.exists():
            sys.exit(f"missing {f} — run from a checkout with provenance/ present")
        d = json.loads(f.read_text())
        crit = sum(1 for r in d.get("results", [])
                   if r.get("category") == "safety_critical" and r.get("score", 1) < 1)
        out.append(dict(score=d["overall"] * 100, crit=crit,
                        l1=l1, l2=l2, phase=phase))
    return out


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def build(pts):
    n = len(pts)
    span = W - L - R
    xs = [L + span * (i + 0.5) / n for i in range(n)]
    y = lambda s: PLOT_B - (s / 100) * (PLOT_B - T)

    s = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
         f'width="{W}" height="{H}" font-family="{FONT}">',
         f'<rect width="{W}" height="{H}" fill="#ffffff"/>']

    # title
    s.append(f'<text x="{L}" y="34" font-size="19" font-weight="700" '
             f'fill="#1f2328">From base model to submission</text>')
    s.append(f'<text x="{L}" y="56" font-size="13" fill="#656d76">'
             f'Every point scored on the same 24-prompt harness, 8 samples per '
             f'prompt. Higher is better.</text>')

    # phase bands behind everything
    for i0, i1, label, tint in BANDS:
        bx0 = L + span * i0 / n + 3
        bx1 = L + span * (i1 + 1) / n - 3
        s.append(f'<rect x="{bx0:.1f}" y="{T-30:.1f}" width="{bx1-bx0:.1f}" '
                 f'height="{PLOT_B-T+74:.1f}" rx="7" fill="{tint}"/>')
        s.append(f'<text x="{(bx0+bx1)/2:.1f}" y="{T-12:.1f}" font-size="12" '
                 f'font-weight="600" fill="#57606a" text-anchor="middle">'
                 f'{esc(label)}</text>')

    s.append(f'<text x="{L}" y="76" font-size="12" fill="#8250df">'
             f'Round 1 shipped a LoRA adapter trained with MLX. It was never run '
             f'against this harness, so it has no point here.</text>')

    # gridlines
    for v in (0, 25, 50, 75, 100):
        gy = y(v)
        width = "1.6" if v == 0 else "1"
        s.append(f'<line x1="{L}" y1="{gy:.1f}" x2="{W-R}" y2="{gy:.1f}" '
                 f'stroke="#d8dee4" stroke-width="{width}"/>')
        s.append(f'<text x="{L-12}" y="{gy+4:.1f}" font-size="11" fill="#656d76" '
                 f'text-anchor="end">{v}%</text>')

    # connecting path through the measured points
    d = " ".join(("M" if i == 0 else "L") + f"{xs[i]:.1f},{y(p['score']):.1f}"
                 for i, p in enumerate(pts))
    s.append(f'<path d="{d}" fill="none" stroke="#afb8c1" stroke-width="2" '
             f'stroke-dasharray="4 4"/>')

    # points
    for x, p in zip(xs, pts):
        col = PHASE[p["phase"]][0]
        py = y(p["score"])
        big = p["phase"] in ("shipped", "rejected")
        if p["phase"] == "shipped":
            s.append(f'<circle cx="{x:.1f}" cy="{py:.1f}" r="19" fill="none" '
                     f'stroke="{col}" stroke-width="2" opacity="0.45"/>')
        s.append(f'<circle cx="{x:.1f}" cy="{py:.1f}" r="{11 if big else 7}" '
                 f'fill="{col}"/>')
        s.append(f'<text x="{x:.1f}" y="{py-20:.1f}" font-size="15" '
                 f'font-weight="700" fill="{col}" text-anchor="middle">'
                 f'{p["score"]:.1f}%</text>')
        s.append(f'<text x="{x:.1f}" y="{PLOT_B+22:.1f}" font-size="12" '
                 f'font-weight="600" fill="#1f2328" text-anchor="middle">'
                 f'{esc(p["l1"])}</text>')
        s.append(f'<text x="{x:.1f}" y="{PLOT_B+38:.1f}" font-size="11" '
                 f'fill="#656d76" text-anchor="middle">{esc(p["l2"])}</text>')

    # callouts on the two that matter
    for x, p in zip(xs, pts):
        if p["phase"] == "shipped":
            s.append(f'<text x="{x:.1f}" y="{y(p["score"])-40:.1f}" font-size="12" '
                     f'font-weight="700" fill="#1a7f37" text-anchor="middle">'
                     f'SHIPPED</text>')
        if p["phase"] == "rejected":
            s.append(f'<text x="{x:.1f}" y="{y(p["score"])-40:.1f}" font-size="12" '
                     f'font-weight="700" fill="#cf222e" text-anchor="middle">'
                     f'REJECTED</text>')

    # safety-critical strip: 5 cells per model, filled = failure
    s.append(f'<text x="{L-12}" y="{SAFETY_Y+18:.1f}" font-size="11" '
             f'fill="#656d76" text-anchor="end">safety</text>')
    s.append(f'<text x="{L-12}" y="{SAFETY_Y+32:.1f}" font-size="11" '
             f'fill="#656d76" text-anchor="end">failures</text>')
    cw, gap = 13, 4
    for x, p in zip(xs, pts):
        x0 = x - (5 * cw + 4 * gap) / 2
        for k in range(5):
            fail = k < p["crit"]
            s.append(f'<rect x="{x0 + k*(cw+gap):.1f}" y="{SAFETY_Y:.1f}" '
                     f'width="{cw}" height="{cw}" rx="2.5" '
                     f'fill="{"#cf222e" if fail else "#d6f0dc"}" '
                     f'stroke="{"#cf222e" if fail else "#aedbba"}"/>')
        s.append(f'<text x="{x:.1f}" y="{SAFETY_Y+32:.1f}" font-size="11" '
                 f'fill="#656d76" text-anchor="middle">{p["crit"]} of 5</text>')

    # legend
    ly = H - 52
    lx = L
    for key in ("base", "single", "two", "shipped", "rejected"):
        col, lab = PHASE[key]
        s.append(f'<circle cx="{lx+6}" cy="{ly-4}" r="6" fill="{col}"/>')
        s.append(f'<text x="{lx+19}" y="{ly}" font-size="12" fill="#1f2328">'
                 f'{esc(lab)}</text>')
        lx += 22 + len(lab) * 7.0
    s.append(f'<text x="{L}" y="{H-22}" font-size="11.5" fill="#656d76">'
             f'Sampling noise is about ±6 points: two byte-identical checkpoints '
             f'scored 75.1% and 81.7% in one sweep.</text>')

    s.append('</svg>')
    return "\n".join(s)


if __name__ == "__main__":
    pts = load()
    out = ROOT / "assets" / "journey.svg"
    out.write_text(build(pts), encoding="utf-8")
    print(f"wrote {out.relative_to(ROOT)}  ({len(pts)} models)")
    for p in pts:
        print(f"  {p['l1']:14} {p['l2']:12} {p['score']:5.1f}%  {p['crit']} crit")
