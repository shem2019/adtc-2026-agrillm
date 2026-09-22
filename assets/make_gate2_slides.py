#!/usr/bin/env python3
"""Three-slide Gate 2 deck in the AgriLLM visual style.

    python3 assets/make_gate2_slides.py  ->  assets/AgriLLM_Gate2.pdf

Every figure is taken from REPORT.md and provenance/benchmark/.
"""
import math
from pathlib import Path

from reportlab.lib.colors import HexColor, Color
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas

W, H = 13.333 * inch, 7.5 * inch
M = 0.95 * inch

BG_TOP = HexColor("#0f2e24")
BG_BOT = HexColor("#08170f")
FG = HexColor("#ffffff")
SOFT = HexColor("#c9d6cf")
MUTED = HexColor("#8fa89b")
GOLD = HexColor("#d9a93a")
LEAF = HexColor("#1f4a36")
LEAF_EDGE = HexColor("#2d6a4d")
CARD = HexColor("#12352a")

BOLD, REG = "Helvetica-Bold", "Helvetica"


def background(c):
    steps = 60
    for i in range(steps):
        t = i / (steps - 1)
        r = BG_BOT.red + (BG_TOP.red - BG_BOT.red) * t
        g = BG_BOT.green + (BG_TOP.green - BG_BOT.green) * t
        b = BG_BOT.blue + (BG_TOP.blue - BG_BOT.blue) * t
        c.setFillColor(Color(r, g, b))
        c.rect(0, H * i / steps, W, H / steps + 1, stroke=0, fill=1)
    # faint contour lines
    c.setStrokeColor(Color(1, 1, 1, alpha=0.05))
    c.setLineWidth(0.8)
    for k in range(9):
        y0 = H * (0.12 + k * 0.1)
        p = c.beginPath()
        p.moveTo(0, y0)
        for x in range(0, int(W) + 20, 20):
            y = y0 + 18 * math.sin(x / 140 + k * 0.7) + 10 * math.sin(x / 57 + k)
            p.lineTo(x, y)
        c.drawPath(p, stroke=1, fill=0)


def leaf(c, x, y, length, angle, width=0.22):
    c.saveState()
    c.translate(x, y)
    c.rotate(angle)
    w = length * width
    p = c.beginPath()
    p.moveTo(0, 0)
    p.curveTo(length * 0.3, w, length * 0.75, w * 0.9, length, 0)
    p.curveTo(length * 0.75, -w * 0.5, length * 0.3, -w * 0.7, 0, 0)
    c.setFillColor(LEAF)
    c.setStrokeColor(LEAF_EDGE)
    c.setLineWidth(1.2)
    c.drawPath(p, stroke=1, fill=1)
    c.setStrokeColor(GOLD)
    c.setLineWidth(1.6)
    c.line(-length * 0.08, 0, length * 0.97, 0)
    c.restoreState()


def rule(c, x, y, w=0.9 * inch):
    c.setFillColor(GOLD)
    c.rect(x, y, w, 5, stroke=0, fill=1)


def kicker(c, text, y):
    c.setFillColor(GOLD)
    c.setFont(BOLD, 14)
    c.drawString(M, y, text.upper())


def title(c, text, y, size=38):
    c.setFillColor(FG)
    c.setFont(BOLD, size)
    c.drawString(M, y, text)


def card(c, x, y, w, h, value, label, sub=None, vcolor=GOLD, vsize=40):
    c.setFillColor(CARD)
    c.roundRect(x, y, w, h, 10, stroke=0, fill=1)
    c.setFillColor(vcolor)
    c.setFont(BOLD, vsize)
    c.drawString(x + 22, y + h - 58, value)
    c.setFillColor(FG)
    c.setFont(BOLD, 16)
    c.drawString(x + 22, y + h - 86, label)
    if sub:
        c.setFillColor(MUTED)
        c.setFont(REG, 13)
        c.drawString(x + 22, y + h - 106, sub)


def footer(c, n):
    c.setFillColor(MUTED)
    c.setFont(REG, 11)
    c.drawString(M, 0.45 * inch, "AgriLLM · Africa Deep Tech Challenge 2026 · Agriculture")
    c.drawRightString(W - M, 0.45 * inch, f"{n} / 3")


def slide_title(c):
    background(c)
    leaf(c, W - 3.2 * inch, H - 1.1 * inch, 4.4 * inch, -128, 0.2)
    leaf(c, W - 1.2 * inch, H - 0.4 * inch, 3.2 * inch, -112, 0.18)
    c.setFillColor(FG)
    c.setFont(BOLD, 96)
    c.drawString(M, H * 0.52, "AgriLLM")
    rule(c, M, H * 0.52 - 34, 1.1 * inch)
    c.setFont(BOLD, 30)
    c.setFillColor(FG)
    c.drawString(M, H * 0.52 - 88, "Offline agricultural intelligence")
    c.setFillColor(GOLD)
    c.drawString(M, H * 0.52 - 128, "for Africa")
    c.setFillColor(SOFT)
    c.setFont(BOLD, 13)
    c.drawString(M, 1.05 * inch,
                 "OFFLINE  ·  ON DEMAND  ·  EVEN ON MODEST HARDWARE  ·  NO SUBSCRIPTION")


def wrap(c, text, x, y, width, font, size, lead, color):
    c.setFillColor(color)
    c.setFont(font, size)
    line = ""
    for word in text.split():
        trial = (line + " " + word).strip()
        if c.stringWidth(trial, font, size) > width:
            c.drawString(x, y, line)
            y -= lead
            line = word
        else:
            line = trial
    if line:
        c.drawString(x, y, line)
        y -= lead
    return y


def slide_journey(c):
    background(c)
    kicker(c, "What it does", H - M)
    title(c, "Advice on demand, in plain words", H - M - 52)
    wrap(c, "African smallholders need farming advice daily, and extension officers are few. "
            "AgriLLM extends their reach by answering everyday farm questions on demand, "
            "offline, on a budget laptop.",
         M, H - M - 92, W - 2 * M, REG, 17, 24, SOFT)
    bw = W - 2 * M
    bh = 190
    by = H - M - 92 - 2 * 24 - 14 - bh
    c.setFillColor(CARD)
    c.roundRect(M, by, bw, bh, 10, stroke=0, fill=1)
    c.setFillColor(GOLD)
    c.setFont(BOLD, 13)
    c.drawString(M + 22, by + bh - 28, "A FARMER ASKS")
    yy = wrap(c, "Small purple flowering weeds are coming up around my maize and the maize is "
                 "stunted even though I applied fertiliser. What is this?",
              M + 22, by + bh - 50, bw - 44, REG, 15, 20, FG)
    c.setFillColor(GOLD)
    c.setFont(BOLD, 13)
    c.drawString(M + 22, yy - 6, "AGRILLM ANSWERS")
    wrap(c, "These are Striga hermonthica, also known as witchweed, which is a parasitic weed. "
            "It attaches to maize roots underground and sucks water and nutrients from the "
            "plant, causing the stunting even with reasonable fertiliser application.",
         M + 22, yy - 28, bw - 44, BOLD, 15, 20, FG)
    items = [
        ("Chickens dying, twisted necks", "names Newcastle disease and the first steps to take"),
        ("Why rotate maize?", "a few plain sentences a farmer can act on"),
        ("A pesticide dose", "points to the product label and the agrodealer"),
    ]
    yy = by - 28
    for k, v in items:
        c.setFillColor(GOLD)
        c.setFont(BOLD, 15)
        c.drawString(M, yy, k)
        c.setFillColor(FG)
        c.setFont(REG, 15)
        c.drawString(M + 3.3 * inch, yy, v)
        yy -= 25
    footer(c, 2)


def slide_measured(c):
    background(c)
    kicker(c, "What fine-tuning achieved", H - M)
    title(c, "Safer, clearer, and it runs anywhere", H - M - 52)
    c.setFillColor(SOFT)
    c.setFont(REG, 17)
    c.drawString(M, H - M - 90, "Against the model it was built on, on a 24-prompt behavioural test")
    cw, ch, gap = 2.65 * inch, 1.6 * inch, 0.25 * inch
    y1 = H - M - 90 - 24 - ch
    card(c, M, y1, cw, ch, "3.5x", "higher overall", "22.7% to 78.9%")
    card(c, M + (cw + gap), y1, cw, ch, "~5x", "right on diagnosis", "and on agronomy")
    card(c, M + 2 * (cw + gap), y1, cw, ch, "up to 8x", "plainer answers", "for non-technical farmers", vsize=36)
    card(c, M + 3 * (cw + gap), y1, cw, ch, "5 to 1", "safety failures", "out of 5 safety prompts", vsize=36)
    c.setFillColor(SOFT)
    c.setFont(REG, 17)
    c.drawString(M, y1 - 34, "Measured with the official ADTC profiler, CPU only, offline")
    y2 = y1 - 44 - ch
    card(c, M, y2, cw, ch, "15.7", "tokens / second", "official profiler image")
    card(c, M + (cw + gap), y2, cw, ch, "1.07 GB", "peak memory", "well within 8 GB")
    card(c, M + 2 * (cw + gap), y2, cw, ch, "~12", "tokens / second", "on an Intel i5-7200U")
    card(c, M + 3 * (cw + gap), y2, cw, ch, "940 MB", "one download", "then fully offline")
    c.setFillColor(MUTED)
    c.setFont(REG, 13)
    c.drawString(M, y2 - 26, "github.com/shem2019/adtc-2026-agrillm  ·  huggingface.co/shemking/agrillm-qwen2.5-1.5b-agri")
    footer(c, 3)


def build(path):
    c = canvas.Canvas(str(path), pagesize=(W, H))
    c.setTitle("AgriLLM — Gate 2")
    slide_title(c)
    c.showPage()
    slide_journey(c)
    c.showPage()
    slide_measured(c)
    c.save()


if __name__ == "__main__":
    out = Path(__file__).with_name("AgriLLM_Gate2.pdf")
    build(out)
    print(out)
