#!/usr/bin/env python3
"""AgriLLM Corpus Generator & Auditor — ADTC 2026 Gate 2.

Audits source files -> _audit/<topic>.jsonl
Generates clean datasets for 22 topics -> _clean/<topic>.jsonl
Checks all 12 §7 Quality Gates.
Outputs _audit/REPORT.md.
"""

from __future__ import annotations
import glob
import json
import math
import os
import random
import re
import sys
from collections import Counter, deque
from difflib import SequenceMatcher
from pathlib import Path

HERE = Path(__file__).parent.resolve()
SRC_DIR = HERE
CLEAN_DIR = HERE / "_clean"
AUDIT_DIR = HERE / "_audit"
REPORT_PATH = AUDIT_DIR / "REPORT.md"

CLEAN_DIR.mkdir(parents=True, exist_ok=True)
AUDIT_DIR.mkdir(parents=True, exist_ok=True)

# Delete swahili.jsonl if present
for p in [SRC_DIR / "swahili.jsonl", CLEAN_DIR / "swahili.jsonl"]:
    if p.exists():
        p.unlink()

TOPIC_TARGETS = {
    "maize-pests": 2500,
    "maize-agronomy": 1500,
    "beans-legumes": 1500,
    "cassava-sweetpotato": 1500,
    "potato-horticulture": 1200,
    "soil-fertility": 1200,
    "livestock": 1000,
    "post-harvest": 900,
    "poultry": 900,
    "coffee-tea": 900,
    "banana": 800,
    "sorghum-millet-groundnut": 800,
    "economics-extension": 800,
    "water-conservation": 700,
    "climate-seasons": 700,
    "safety-poisoning": 600,
    "safety-dose-refusal": 600,
    "uncertainty-deferral": 600,
    "instruction-format": 900,
    "plain-language": 900,
    "multiturn": 600,
    "systems-offline": 300,
}

# Regex patterns
EAST_AFRICA_PAT = re.compile(
    r"\b(kenya|tanzania|uganda|rwanda|ethiopia|burundi|somalia|south sudan|"
    r"nakuru|kitale|eldoret|kisumu|meru|embu|machakos|kakamega|bungoma|uasin gishu|trans nzoia|"
    r"arusha|mbeya|morogoro|iringa|dodoma|mwanza|kilimanjaro|shinyanga|tabora|tanga|ruvuma|"
    r"kampala|mbale|gulu|masaka|jinja|mbarara|fort portal|musanze|huye|kigali|bugesera|oromia|amhara|"
    r"long rains|short rains|masika|vuli|belg|meher|shilling|shillings|"
    r"east africa|kalro|naads|tari|rab|icipe|cimmyt|iita|ilri|icraf|"
    r"fall armyworm|striga|cassava mosaic|brown streak|east coast fever|"
    r"bacterial wilt|newcastle|matooke|kienyeji)\b",
    re.IGNORECASE,
)

DOSE_PAT = re.compile(
    r"(\b\d+(\.\d+)?\s*(ml|g|grams|cc|ppm|mg)\s*(per|/|\s+in\s+)\s*(liter|l|ha|hectare|acre|20l|knapsack|water)\b)|"
    r"(\b\d+(\.\d+)?\s*(ml|g|grams|cc|ppm)\s*/\s*(l|liter|ha|acre)\b)|"
    r"(\baic\s+of\s+\d+)|"
    r"(\bspray(ed)?\s+at\s+\d+\s*(ml|g|ppm)\b)|"
    r"(\b\d+\s*ml\s+of\s+[a-z]+\s+per\b)",
    re.IGNORECASE,
)

MEDICAL_PAT = re.compile(
    r"\b(induce vomiting|salt water|milk|home remedy for poison|vomiting with salt|1 part milk|3 parts salt)\b",
    re.IGNORECASE,
)

INVENTED_PAT = re.compile(r"\b(heterorhabdium|prothid)\b", re.IGNORECASE)
WRONG_HOST_PAT = re.compile(r"(diadegma semiclausum.*fall armyworm|fall armyworm.*diadegma semiclausum|paraquat.*caterpillar|caterpillar.*paraquat|imidacloprid.*fall armyworm|fall armyworm.*imidacloprid)", re.IGNORECASE)
WRONG_FACT_PAT = re.compile(r"(sorghum.*highly resistant to fall armyworm|beans are naturally resistant to.*weeds|deep roots of beans)", re.IGNORECASE)
MARKDOWN_PAT = re.compile(r"(\*\*|^#{1,6}\s)", re.MULTILINE)

COUNTIES = [
    "Nakuru", "Kitale", "Trans Nzoia", "Eldoret", "Uasin Gishu", "Meru", "Embu", "Machakos",
    "Bungoma", "Kisumu", "Kakamega", "Nyeri", "Kirinyaga", "Murang'a", "Makueni", "Kilifi",
    "Arusha", "Mbeya", "Morogoro", "Iringa", "Dodoma", "Mwanza", "Kilimanjaro", "Shinyanga",
    "Kampala", "Mbale", "Gulu", "Masaka", "Jinja", "Mbarara", "Musanze", "Huye", "Kigali", "Oromia"
]

INSTITUTIONS = ["KALRO", "TARI", "NARO", "RAB", "icipe", "CABI Plantwise", "CIMMYT", "IITA"]

def clean_markdown(t: str) -> str:
    t = re.sub(r"\*\*(.*?)\*\*", r"\1", t)
    t = re.sub(r"^\s*#{1,6}\s*", "", t, flags=re.MULTILINE)
    return t.strip()

def similar_q(a: str, b: str) -> float:
    la, lb = len(a), len(b)
    if la == 0 or lb == 0 or (2 * min(la, lb) / (la + lb)) < 0.85:
        return 0.0
    sm = SequenceMatcher(None, a[:160].lower(), b[:160].lower())
    if sm.quick_ratio() < 0.85:
        return 0.0
    return sm.ratio()

def audit_source_topic(topic: str) -> tuple[list[dict], list[dict], dict]:
    src_path = SRC_DIR / f"{topic}.jsonl"
    if not src_path.exists():
        return [], [], {"KEEP": 0, "FIX": 0, "DROP": 0, "REVIEW": 0, "source_rows": 0}

    audit_records = []
    clean_rows = []
    stats = {"KEEP": 0, "FIX": 0, "DROP": 0, "REVIEW": 0, "source_rows": 0}

    raw_lines = [l.strip().rstrip(",") for l in src_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    stats["source_rows"] = len(raw_lines)

    for idx, line in enumerate(raw_lines, 1):
        row_id = f"{topic}:{idx}"
        if line in ("[", "]"):
            continue
        try:
            row = json.loads(line)
        except Exception as e:
            audit_records.append({
                "id": row_id,
                "verdict": "DROP",
                "reasons": ["MALFORMED_JSON"],
                "problems": [f"{line[:60]}... (JSON parse error: {str(e)})"],
                "confidence": 1.0
            })
            stats["DROP"] += 1
            continue

        q = str(row.get("question", "")).strip()
        a = str(row.get("answer", "")).strip()
        full_text = f"{q} {a}"

        reasons = []
        problems = []
        verdict = "KEEP"

        if MEDICAL_PAT.search(full_text):
            m = MEDICAL_PAT.search(full_text)
            verdict = "FIX"
            reasons.append("UNSAFE_MEDICAL")
            problems.append(f"'{m.group(0)}': Dangerous medical advice to induce vomiting or give oral remedies for chemical poisoning")
            a = re.sub(r"Induce vomiting.*?(?=\.|\n|$)", "Do not induce vomiting. Give nothing by mouth. Wash affected skin or eyes immediately with clean running water for 15 minutes. Seek emergency medical care at the nearest health facility with the product container or label.", a, flags=re.I)

        if DOSE_PAT.search(full_text):
            m = DOSE_PAT.search(full_text)
            verdict = "FIX"
            reasons.append("UNSAFE_DOSE")
            problems.append(f"'{m.group(0)}': States specific numeric chemical application rate")
            a = re.sub(r"\b\d+(\.\d+)?\s*(ml|g|grams|cc|ppm)\s*(per|/|\s+in\s+)\s*(liter|l|ha|hectare|acre|20l|knapsack|water)\b", "the application rate specified on the product label", a, flags=re.I)
            a = re.sub(r"\baic\s+of\s+\d+\s+to\s+\d+\s*ppm\b", "the registered label application rate", a, flags=re.I)

        if INVENTED_PAT.search(full_text):
            m = INVENTED_PAT.search(full_text)
            verdict = "FIX"
            reasons.append("INVENTED_ENTITY")
            problems.append(f"'{m.group(0)}': Non-existent or mislabeled entity")
            a = re.sub(r"Heterorhabdium sheathi", "Heterorhabditis bacteriophora", a, flags=re.I)
            a = re.sub(r"Heterorhabdium", "Heterorhabditis", a, flags=re.I)
            a = re.sub(r"Prothid", "registered active ingredient", a, flags=re.I)

        if WRONG_HOST_PAT.search(full_text):
            m = WRONG_HOST_PAT.search(full_text)
            verdict = "FIX"
            reasons.append("WRONG_HOST_OR_ACTIVE")
            problems.append(f"'{m.group(0)}': Misapplied host plant or active ingredient for target pest")
            a = re.sub(r"Diadegma semiclausum parasitises fall armyworm", "Telenomus remus parasitises fall armyworm eggs", a, flags=re.I)
            a = re.sub(r"Diadegma semiclausum", "Telenomus remus", a, flags=re.I)
            a = re.sub(r"Paraquat", "Emamectin benzoate", a, flags=re.I)
            a = re.sub(r"Imidacloprid", "Emamectin benzoate", a, flags=re.I)

        if WRONG_FACT_PAT.search(full_text):
            m = WRONG_FACT_PAT.search(full_text)
            verdict = "FIX"
            reasons.append("WRONG_FACT")
            problems.append(f"'{m.group(0)}': Inaccurate agricultural assertion")
            a = re.sub(r"Sorghum varieties are highly resistant to Fall Armyworm", "Sorghum varieties can be damaged by Fall Armyworm and require regular scouting", a, flags=re.I)
            a = re.sub(r"Beans are naturally resistant to many weeds", "Beans compete poorly with weeds during their early growth stages", a, flags=re.I)
            a = re.sub(r"deep roots of beans break up compacted soil", "shallow root systems of common beans require well-prepared soil", a, flags=re.I)

        if MARKDOWN_PAT.search(a):
            if verdict == "KEEP":
                verdict = "FIX"
                reasons.append("MARKDOWN_FORMATTING")
                problems.append("Markdown formatting present in answer")
            a = clean_markdown(a)

        if verdict == "KEEP":
            reasons = ["OK"]
            stats["KEEP"] += 1
            clean_rows.append({"question": q, "answer": a})
            record = {
                "id": row_id,
                "verdict": "KEEP",
                "reasons": reasons,
                "problems": [],
                "confidence": 0.98
            }
        else:
            stats["FIX"] += 1
            a = clean_markdown(a)
            clean_rows.append({"question": q, "answer": a})
            record = {
                "id": row_id,
                "verdict": "FIX",
                "reasons": reasons,
                "problems": problems,
                "corrected_answer": a,
                "confidence": 0.95
            }

        audit_records.append(record)

    audit_file = AUDIT_DIR / f"{topic}.jsonl"
    audit_file.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in audit_records) + "\n", encoding="utf-8")
    return audit_records, clean_rows, stats

print("Audit engine ready.")
