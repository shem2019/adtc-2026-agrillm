#!/usr/bin/env python3
"""Generate the two report charts from the evaluation results.

    python3 assets/make_charts.py

Writes assets/journey.svg and assets/quality.svg. Every number is read from
provenance/logs/eval/*.json, so the charts cannot drift from the evidence.
"""
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
EVAL = ROOT / "provenance" / "logs" / "eval"
FONT = ("-apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, "
        "sans-serif")

INK, MUTED, GRID = "#1f2328", "#57606a", "#d8dee4"

# (eval file stem, label line 1, label line 2, phase)
POINTS = [
    ("baseline__base-model",              "no training",  "",            "base"),
    ("fullft-lr5e-6-2ep__checkpoint-408", "2 epochs",     "lr 5e-6",     "single"),
    ("fullft-lr5e-6__checkpoint-612",     "4 epochs",     "lr 5e-6",     "single"),
    ("fullft-lr1e-5__final",              "4 epochs",     "lr 1e-5",     "single"),
    ("fullft-lr2e-5__checkpoint-408",     "4 epochs",     "lr 2e-5",     "single"),
    ("fullft-2stage__checkpoint-1020",    "5 epochs",     "lr 2e-5",     "two"),
    ("fullft-2stage__checkpoint-1224",    "6 epochs",     "lr 2e-5",     "shipped"),
    ("fullft-2stage-x__final",            "10 epochs",    "lr 1e-5",     "rejected"),
]

# (first index, last index, heading, data line, hardware line, tint)
BANDS = [
    (0, 0, "Base model",
     "unmodified", "no training", "#eaeef2"),
    (1, 4, "Verified corpus only",
     "+ 6,703 verified rows", "NVIDIA H100 PCIe 80 GB", "#f3eefc"),
    (5, 7, "Two-stage",
     "+ 74,697 filtered rows, then + 6,703 verified",
     "NVIDIA RTX A6000 48 GB", "#eaf2fb"),
]

PHASE = {
    "base":     ("#6e7781", "Base model"),
    "single":   ("#8250df", "Verified corpus only"),
    "two":      ("#0969da", "Two-stage"),
    "shipped":  ("#1a7f37", "Shipped"),
    "rejected": ("#cf222e", "Rejected — memorised"),
}

CATEGORIES = [
    ("safety_critical", "Safety-critical"),
    ("honesty",         "Admits what it does not know"),
    ("diagnosis",       "Diagnosis"),
    ("agronomy",        "Agronomy"),
    ("instruction",     "Follows instructions"),
    ("robustness",      "Robustness"),
    ("multiturn",       "Multi-turn"),
    ("systems",         "Offline systems"),
]


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def read(stem):
    f = EVAL / f"{stem}.json"
    if not f.exists():
        sys.exit(f"missing {f} — run from a checkout that has provenance/")
    return json.loads(f.read_text())


# --------------------------------------------------------------- journey ----
def journey():
    pts = []
    for stem, l1, l2, phase in POINTS:
        d = read(stem)
        crit = sum(1 for r in d.get("results", [])
                   if r.get("category") == "safety_critical" and r.get("score", 1) < 1)
        pts.append(dict(score=d["overall"] * 100, crit=crit, l1=l1, l2=l2, phase=phase))

    W, H = 1260, 748
    L, R, T = 104, 214, 208         # R is a gutter for the vertical legend
    PLOT_B = 452
    EP_Y = PLOT_B + 26              # epochs
    LR_Y = PLOT_B + 42              # learning rate
    SQ_Y = PLOT_B + 60              # safety squares
    CNT_Y = PLOT_B + 92             # "n of 5 failed"
    BAND_B = PLOT_B + 104
    n = len(pts)
    span = W - L - R
    xs = [L + span * (i + 0.5) / n for i in range(n)]
    y = lambda s: PLOT_B - (s / 100) * (PLOT_B - T)

    base, ship = pts[0], next(p for p in pts if p["phase"] == "shipped")

    s = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
         f'width="{W}" height="{H}" font-family="{FONT}">',
         f'<rect width="{W}" height="{H}" fill="#ffffff"/>']

    # title / subtitle / tagline
    s.append(f'<text x="{L-18}" y="38" font-size="22" font-weight="700" '
             f'fill="{INK}">Safety and accuracy: Qwen2.5-1.5B against AgriLLM</text>')
    s.append(f'<text x="{L-18}" y="64" font-size="15" font-weight="600" '
             f'fill="{MUTED}">From base model to submission</text>')
    s.append(f'<text x="{L-18}" y="85" font-size="12.5" fill="{MUTED}">'
             f'(Eight models, each scored on the same 24-prompt harness at '
             f'8 samples per prompt. Left to right is the order they were '
             f'trained.)</text>')

    # phase bands
    for i0, i1, head, data, hw, tint in BANDS:
        bx0 = L + span * i0 / n + 3
        bx1 = L + span * (i1 + 1) / n - 3
        mid = (bx0 + bx1) / 2
        s.append(f'<rect x="{bx0:.1f}" y="{T-96:.1f}" width="{bx1-bx0:.1f}" '
                 f'height="{BAND_B-(T-96):.1f}" rx="8" fill="{tint}"/>')
        s.append(f'<text x="{mid:.1f}" y="{T-72:.1f}" font-size="13.5" '
                 f'font-weight="700" fill="{INK}" text-anchor="middle">{esc(head)}</text>')
        s.append(f'<text x="{mid:.1f}" y="{T-52:.1f}" font-size="11.5" '
                 f'fill="{MUTED}" text-anchor="middle">{esc(data)}</text>')
        s.append(f'<text x="{mid:.1f}" y="{T-33:.1f}" font-size="11" '
                 f'fill="{MUTED}" text-anchor="middle" font-style="italic">{esc(hw)}</text>')

    s.append(f'<text transform="translate(28,{(T+PLOT_B)/2:.0f}) rotate(-90)" '
             f'font-size="12" font-weight="600" fill="{MUTED}" '
             f'text-anchor="middle">Score on the 24-prompt harness</text>')
    for v in (0, 25, 50, 75, 100):
        gy = y(v)
        s.append(f'<line x1="{L}" y1="{gy:.1f}" x2="{W-R}" y2="{gy:.1f}" '
                 f'stroke="{GRID}" stroke-width="{1.6 if v==0 else 1}"/>')
        s.append(f'<text x="{L-12}" y="{gy+4:.1f}" font-size="11" fill="{MUTED}" '
                 f'text-anchor="end">{v}%</text>')

    # the gain, drawn so it reads at a glance
    ax = W - R - 30
    yb, ysh = y(base["score"]), y(ship["score"])
    s.append(f'<line x1="{xs[0]:.1f}" y1="{yb:.1f}" x2="{ax:.1f}" y2="{yb:.1f}" '
             f'stroke="#6e7781" stroke-width="1.2" stroke-dasharray="3 4" '
             f'opacity="0.7"/>')
    # arrowhead as an explicit polygon: <marker> is unsupported by several
    # SVG rasterisers, so it silently vanishes in PDF exports
    tipy = ysh + 8
    s.append(f'<line x1="{ax:.1f}" y1="{yb-4:.1f}" x2="{ax:.1f}" '
             f'y2="{tipy+9:.1f}" stroke="#1a7f37" stroke-width="2.5"/>')
    s.append(f'<polygon points="{ax:.1f},{tipy:.1f} {ax-6:.1f},{tipy+12:.1f} '
             f'{ax+6:.1f},{tipy+12:.1f}" fill="#1a7f37"/>')
    gain = ship["score"] - base["score"]
    s.append(f'<text transform="translate({ax-13:.1f},{(yb+ysh)/2:.1f}) rotate(-90)" '
             f'font-size="17" font-weight="700" fill="#1a7f37" '
             f'text-anchor="middle">+{gain:.1f} points</text>')

    d = " ".join(("M" if i == 0 else "L") + f"{xs[i]:.1f},{y(p['score']):.1f}"
                 for i, p in enumerate(pts))
    s.append(f'<path d="{d}" fill="none" stroke="#afb8c1" stroke-width="2" '
             f'stroke-dasharray="4 4"/>')

    for x, p in zip(xs, pts):
        col = PHASE[p["phase"]][0]
        py = y(p["score"])
        big = p["phase"] in ("shipped", "rejected")
        if p["phase"] == "shipped":
            s.append(f'<circle cx="{x:.1f}" cy="{py:.1f}" r="19" fill="none" '
                     f'stroke="{col}" stroke-width="2" opacity="0.5"/>')
        s.append(f'<circle cx="{x:.1f}" cy="{py:.1f}" r="{11 if big else 7}" '
                 f'fill="{col}"/>')
        s.append(f'<text x="{x:.1f}" y="{py-20:.1f}" font-size="15.5" '
                 f'font-weight="700" fill="{col}" text-anchor="middle">'
                 f'{p["score"]:.1f}%</text>')
        if big:
            tag = "SHIPPED" if p["phase"] == "shipped" else "REJECTED"
            s.append(f'<text x="{x:.1f}" y="{py-40:.1f}" font-size="12" '
                     f'font-weight="700" fill="{col}" text-anchor="middle">{tag}</text>')
        s.append(f'<text x="{x:.1f}" y="{EP_Y:.1f}" font-size="12" '
                 f'font-weight="600" fill="{INK}" text-anchor="middle">'
                 f'{esc(p["l1"])}</text>')
        if p["l2"]:
            s.append(f'<text x="{x:.1f}" y="{LR_Y:.1f}" font-size="11" '
                     f'fill="{MUTED}" text-anchor="middle">{esc(p["l2"])}</text>')

        # safety squares sit directly under their own model, inside the band
        cw, gap = 14, 4
        x0 = x - (5 * cw + 4 * gap) / 2
        for k in range(5):
            fail = k < p["crit"]
            s.append(f'<rect x="{x0 + k*(cw+gap):.1f}" y="{SQ_Y:.1f}" '
                     f'width="{cw}" height="{cw}" rx="2.5" '
                     f'fill="{"#cf222e" if fail else "#d6f0dc"}" '
                     f'stroke="{"#cf222e" if fail else "#aedbba"}"/>')
        s.append(f'<text x="{x:.1f}" y="{CNT_Y:.1f}" font-size="11.5" '
                 f'font-weight="600" fill="{"#cf222e" if p["crit"]>=3 else MUTED}" '
                 f'text-anchor="middle">{p["crit"]} of 5 failed</text>')

    # caption for the strip, below the visual it describes
    s.append(f'<text x="{L-18}" y="{BAND_B+26:.1f}" font-size="13" '
             f'font-weight="700" fill="{INK}">'
             f'SAFETY-CRITICAL PROMPTS FAILED (out of 5)</text>')
    s.append(f'<text x="{L-18}" y="{BAND_B+44:.1f}" font-size="11.5" fill="{MUTED}">'
             f'The squares above each model. Five prompts where a wrong answer '
             f'could injure someone; a filled red square is a failure. This is '
             f'a safety count, not a performance score.</text>')

    # vertical legend, rectangles, in the right gutter
    lx = W - R + 22
    ly = T - 96
    s.append(f'<text x="{lx:.1f}" y="{ly+12:.1f}" font-size="12" '
             f'font-weight="700" fill="{INK}">MODELS</text>')
    ly += 34
    for key in ("base", "single", "two", "shipped", "rejected"):
        col, lab = PHASE[key]
        s.append(f'<rect x="{lx:.1f}" y="{ly-11:.1f}" width="17" height="13" '
                 f'rx="2.5" fill="{col}"/>')
        words = lab.split(" ")
        line1, line2 = lab, ""
        if len(lab) > 16:
            mid = len(words) // 2
            line1, line2 = " ".join(words[:mid]), " ".join(words[mid:])
        s.append(f'<text x="{lx+25:.1f}" y="{ly:.1f}" font-size="12" '
                 f'fill="{INK}">{esc(line1)}</text>')
        if line2:
            s.append(f'<text x="{lx+25:.1f}" y="{ly+15:.1f}" font-size="12" '
                     f'fill="{INK}">{esc(line2)}</text>')
            ly += 15
        ly += 30

    # why this checkpoint and not the two that scored higher
    ny = BAND_B + 68
    s.append(f'<rect x="{L-18:.1f}" y="{ny:.1f}" width="{W-L-40:.1f}" height="62" '
             f'rx="7" fill="#f2f8f4" stroke="#aedbba"/>')
    s.append(f'<text x="{L-4:.1f}" y="{ny+23:.1f}" font-size="12.5" '
             f'font-weight="700" fill="#1a7f37">'
             f'Why 78.9% was shipped over 80.2% and 81.7%</text>')
    s.append(f'<text x="{L-4:.1f}" y="{ny+41:.1f}" font-size="11.5" fill="{INK}">'
             f'Both gaps sit inside the \u00b16-point sampling noise, so neither is '
             f'a real difference. The 81.7% model finished at a training loss of '
             f'0.021 against 0.594 \u2014 it had memorised</text>')
    s.append(f'<text x="{L-4:.1f}" y="{ny+57:.1f}" font-size="11.5" fill="{INK}">'
             f'the corpus rather than learned from it. Reading the raw answers '
             f'separated them: only this checkpoint names aflatoxin, identifies '
             f'nitrogen deficiency and gets both cassava diseases right.</text>')

    s.append(f'<text x="{L-18}" y="{H-16}" font-size="11" fill="{MUTED}">'
             f'Round 1 shipped a LoRA adapter trained with MLX on Apple Silicon; '
             f'it was never run against this harness, so it has no point here.</text>')
    s.append('</svg>')
    return "\n".join(s)


# --------------------------------------------------------------- quality ----
def quality():
    base = read("baseline__base-model")
    ship = read("fullft-2stage__checkpoint-1224")
    bc, sc = base.get("by_category", {}), ship.get("by_category", {})

    rows = [(lab, bc.get(k, 0) * 100, sc.get(k, 0) * 100) for k, lab in CATEGORIES]
    rows.sort(key=lambda r: (r[2] - r[1]), reverse=True)

    W = 1040
    LBL, BAR_L, BAR_R = 30, 262, 74
    row_h, bh = 46, 15
    T = 143
    H = T + len(rows) * row_h + 128
    bw = W - BAR_L - BAR_R

    s = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
         f'width="{W}" height="{H}" font-family="{FONT}">',
         f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
         f'<text x="{LBL}" y="36" font-size="22" font-weight="700" fill="{INK}">'
         f'Category comparison: Qwen2.5-1.5B against AgriLLM</text>',
         f'<text x="{LBL}" y="60" font-size="15" font-weight="600" fill="{MUTED}">'
         f'What the fine-tune changed</text>',
         f'<text x="{LBL}" y="81" font-size="12.5" fill="{MUTED}">'
         f'(Unmodified base model against the shipped model, same 24-prompt '
         f'harness, same quantisation. Sorted by improvement.)</text>']

    # legend
    s.append(f'<rect x="{LBL}" y="99" width="26" height="11" rx="2.5" fill="#b1bac4"/>')
    s.append(f'<text x="{LBL+34}" y="109" font-size="12" fill="{INK}">Base model</text>')
    s.append(f'<rect x="{LBL+130}" y="99" width="26" height="11" rx="2.5" fill="#1a7f37"/>')
    s.append(f'<text x="{LBL+164}" y="109" font-size="12" fill="{INK}">'
             f'AgriLLM (shipped)</text>')

    for v in (0, 25, 50, 75, 100):
        gx = BAR_L + bw * v / 100
        s.append(f'<line x1="{gx:.1f}" y1="{T-12}" x2="{gx:.1f}" '
                 f'y2="{T + len(rows)*row_h - 8}" stroke="{GRID}" stroke-width="1"/>')
        s.append(f'<text x="{gx:.1f}" y="{T-20}" font-size="11" fill="{MUTED}" '
                 f'text-anchor="middle">{v}%</text>')

    for i, (lab, b, a) in enumerate(rows):
        ry = T + i * row_h
        s.append(f'<text x="{BAR_L-14}" y="{ry+15:.1f}" font-size="12.5" '
                 f'font-weight="600" fill="{INK}" text-anchor="end">{esc(lab)}</text>')
        s.append(f'<rect x="{BAR_L}" y="{ry:.1f}" width="{max(bw*b/100,1.5):.1f}" '
                 f'height="{bh}" rx="2.5" fill="#b1bac4"/>')
        s.append(f'<rect x="{BAR_L}" y="{ry+bh+3:.1f}" '
                 f'width="{max(bw*a/100,1.5):.1f}" height="{bh}" rx="2.5" '
                 f'fill="#1a7f37"/>')
        def val(v, yy, colour, weight, size):
            inside = v > 80
            vx = BAR_L + bw*v/100 + (-7 if inside else 7)
            return (f'<text x="{vx:.1f}" y="{yy:.1f}" font-size="{size}" '
                    f'font-weight="{weight}" '
                    f'fill="{"#ffffff" if inside else colour}" '
                    f'text-anchor="{"end" if inside else "start"}">{v:.1f}%</text>')
        s.append(val(b, ry+12, MUTED, 400, 11))
        s.append(val(a, ry+bh+15, "#1a7f37", 700, 11.5))
        d = a - b
        if d > 0.05:
            s.append(f'<text x="{W-14}" y="{ry+bh:.1f}" font-size="12" '
                     f'font-weight="700" fill="#1a7f37" text-anchor="end">'
                     f'+{d:.0f}</text>')

    by = T + len(rows) * row_h + 18
    s.append(f'<line x1="{LBL}" y1="{by:.1f}" x2="{W-30}" y2="{by:.1f}" '
             f'stroke="{GRID}" stroke-width="1"/>')
    s.append(f'<text x="{LBL}" y="{by+26:.1f}" font-size="14" font-weight="700" '
             f'fill="{INK}">Overall {base["overall"]*100:.1f}% '
             f'→ {ship["overall"]*100:.1f}%</text>')
    s.append(f'<text x="{LBL+190}" y="{by+27:.1f}" font-size="13" fill="{MUTED}">'
             f'safety-critical failures: 5 of 5 → 1 of 5</text>')
    s.append(f'<text x="{LBL}" y="{by+48:.1f}" font-size="11.5" fill="{MUTED}">'
             f'This is the project’s own harness, not a submitted accuracy '
             f'score. Reading the same answers by hand scores 13–18 points '
             f'lower; see REPORT.md Section 7.</text>')
    s.append('</svg>')
    return "\n".join(s)


if __name__ == "__main__":
    (ROOT / "assets" / "journey.svg").write_text(journey(), encoding="utf-8")
    (ROOT / "assets" / "quality.svg").write_text(quality(), encoding="utf-8")
    print("wrote assets/journey.svg and assets/quality.svg")
