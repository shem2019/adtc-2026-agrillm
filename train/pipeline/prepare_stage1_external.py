#!/usr/bin/env python3
"""Filter AI71ai/agrillm-train-146k into stage-1 training shards.

    python3 train/pipeline/prepare_stage1_external.py \
        --download --out-dir train/external/_stage1

    python3 train/pipeline/prepare_stage1_external.py \
        --source ~/agrillm-train-146k/train.jsonl --out-dir train/external/_stage1

Stage 1 of a two-stage run: broad agricultural knowledge from a third-party
corpus, before stage 2 specialises on our own 6,703 verified rows.

Why this corpus needs filtering rather than using as-is. Its own dataset card
says it is "not authoritative", that "synthetic and curated data may contain
inaccuracies", and recommends "expert validation for any downstream deployment
affecting farmers". Roughly 86% of it is LLM-generated. A sample of it showed:

  * ~28% of rows carry no agricultural vocabulary at all - one sampled row was
    about setting up a garment factory, another was a leftover instruction-
    tuning artefact answering a question about a board game
  * ~3% is non-English (Hindi, French)
  * rows that state pesticide doses flatly, including WHO Class Ib actives like
    Carbofuran, with no PPE, no re-entry interval, no label reference
  * only ~8% mentions East Africa at all

So: drop the off-topic and non-English rows, and quarantine anything carrying a
treatment dose. The dose filter reuses verify_corpus.py's own DOSE_RATE, which
already distinguishes a pesticide rate (banned) from a fertiliser rate
(allowed), so stage 1 cannot teach the model the exact habit stage 2 exists to
suppress.

Output is a directory of {"question","answer"} JSONL shards that
prepare_sft_data.py consumes unchanged - same chat template, same loss mask,
same system-prompt mix as our own corpus.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO / "train"))
from verify_corpus import BANNED_ABSTRACT, DOSE_RATE, GEN_TAG, MARKDOWN  # noqa: E402

HF_DATASET = "AI71ai/agrillm-train-146k"

AGRI_TERMS = re.compile(
    r'\b(crop|farm|soil|seed|fertiliz|fertilis|livestock|poultry|cattle|goat|sheep|'
    r'pig(let)?|chicken|hen|pest|weed|harvest|yield|irrigat|agricultur|agronom|'
    r'plant(ing|ed|s)?\b|cultivat|manure|compost|greenhouse|maize|wheat|rice\b|bean|'
    r'cassava|banana|coffee|tea\b|dairy|veterinar|agroforest|drought|rainfall|season|'
    r'orchard|grain|tuber|vegetable|nursery|hectare|acre|paddy|sorghum|millet|'
    r'groundnut|potato|tomato|onion|cabbage|mulch|pasture|grazing)\b', re.I)

EA_TERMS = re.compile(
    r'\b(kenya|tanzania|uganda|ethiopia|rwanda|burundi|somalia|south sudan|malawi|'
    r'zambia|kalro|nairobi|kampala|dodoma|arusha|kigali|kisumu|mombasa|'
    r'dar es salaam)\b', re.I)

DEVANAGARI = re.compile(r'[ऀ-ॿ]')
FRENCH = re.compile(r'\b(le|la|les|des|une|et|pour|avec|engrais|fumure|cultures?)\b', re.I)

# Extra dose shapes the main regex does not cover, found by reading the corpus:
# "20 cartloads/hectare", "2-3 tons per acre", "NPK 15-15-15".
EXTRA_DOSE = re.compile(
    r'\b\d+\s*(?:-\s*\d+\s*)?(?:tons?|tonnes?|cartloads?|bags?|kg|litres?|liters?|ml|g)\s*'
    r'(?:per|/)\s*(?:acre|hectare|ha|feddan|plant|tree|animal|bird|head)\b'
    r'|\bNPK\s*\d+\s*[-:]\s*\d+\s*[-:]\s*\d+', re.I)


# The corpus's "user" turns are not questions - they are the generation harness's
# own prompt wrapper with the real question embedded:
#
#   "Generate a well crafted, insightful, suitable and correct response to the
#    given question\n\n### QUESTION:\nHow do perennial vegetables..."
#
# Training on that teaches the model that a user message looks like
# "Generate a well crafted... ### QUESTION:", which is nothing like how a grader
# or a farmer asks. Strip it back to the actual question.
# The full wrapper is:
#     <meta instruction>\n\n### QUESTION:\n<the real question>\n\n### Answer:
# so the real question has to be cut out of the middle - taking everything after
# "### QUESTION:" leaves the trailing "### Answer:" attached.
QUESTION_MARKER = re.compile(r'###\s*(?:QUESTION|Q)\s*:?\s*', re.I)
ANSWER_MARKER = re.compile(r'###\s*(?:ANSWER|A|RESPONSE|REPLY)\s*:?.*$', re.I | re.S)
META_INSTRUCTION = re.compile(
    r'^\s*(?:generate|develop|please create|create|write|provide|craft|compose)\b'
    r'[^\n]{0,200}?(?:response|answer|reply)\b[^\n]{0,200}\n+', re.I)
LEADING_HASHES = re.compile(r'^\s*#{1,6}\s*')
ANY_HASH_HEADER = re.compile(r'#{2,}')


def clean_question(q: str) -> str:
    if QUESTION_MARKER.search(q):
        q = QUESTION_MARKER.split(q)[-1]
    else:
        q = META_INSTRUCTION.sub("", q)
    q = ANSWER_MARKER.sub("", q)
    q = LEADING_HASHES.sub("", q).strip()
    return q


def get_text(row: dict) -> tuple[str, str]:
    """The corpus stores assistant sometimes as a string, sometimes as a list."""
    turns = row.get("turns") or []
    if not turns:
        return "", ""
    t = turns[0]
    q = clean_question(str(t.get("user", "")))
    a = t.get("assistant", "")
    if isinstance(a, list):
        a = "\n".join(str(x) for x in a)
    return q, str(a).strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, help="local train.jsonl")
    ap.add_argument("--download", action="store_true", help="fetch from the Hub instead")
    ap.add_argument("--out-dir", type=Path, default=REPO / "train/external/_stage1")
    ap.add_argument("--shard-size", type=int, default=10000)
    ap.add_argument("--max-rows", type=int, default=None, help="cap, for a smoke test")
    ap.add_argument("--min-answer-words", type=int, default=15)
    ap.add_argument("--keep-non-ea", action="store_true", default=True,
                    help="keep general global agronomy, not only East Africa")
    args = ap.parse_args()

    src = args.source
    if args.download or src is None:
        from huggingface_hub import hf_hub_download
        print(f"downloading {HF_DATASET} ...")
        src = Path(hf_hub_download(HF_DATASET, "train.jsonl", repo_type="dataset"))
    if not src.exists():
        print(f"source not found: {src}", file=sys.stderr)
        return 1
    print(f"source: {src}  ({src.stat().st_size / 2**20:.0f} MB)")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stats = Counter()
    kept: list[dict] = []
    quarantined: list[dict] = []

    with src.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            stats["total"] += 1
            if args.max_rows and stats["total"] > args.max_rows:
                break
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                stats["bad_json"] += 1
                continue

            q, a = get_text(row)
            if not q or not a:
                stats["empty"] += 1
                continue
            # A question left as scaffolding, or reduced to nothing by stripping
            # it, is not a question.
            if (len(q.split()) < 4 or QUESTION_MARKER.search(q)
                    or META_INSTRUCTION.search(q) or ANY_HASH_HEADER.search(q)):
                stats["scaffolding_only"] += 1
                continue

            blob = f"{q} {a}"

            if DEVANAGARI.search(blob) or len(FRENCH.findall(blob)) >= 4:
                stats["non_english"] += 1
                continue
            if not AGRI_TERMS.search(blob):
                stats["off_topic"] += 1
                continue
            if len(a.split()) < args.min_answer_words:
                stats["too_short"] += 1
                continue

            # Safety quarantine. Stage 1 must not teach dose-stating, because
            # stage 2's whole job is refusing it - 167 rows of refusal will not
            # reliably undo tens of thousands of examples of compliance.
            if DOSE_RATE.search(a) or EXTRA_DOSE.search(a):
                stats["dose_quarantined"] += 1
                quarantined.append({"question": q, "answer": a})
                continue
            if BANNED_ABSTRACT.search(a):
                stats["banned_abstraction"] += 1
                continue
            if GEN_TAG.search(a):
                stats["generation_tag"] += 1
                continue

            a = MARKDOWN.sub("", a) if MARKDOWN.search(a) else a
            stats["kept"] += 1
            stats["kept_east_africa"] += 1 if EA_TERMS.search(blob) else 0
            kept.append({"question": q, "answer": a})

    print("\nfilter results")
    for k in ("total", "bad_json", "empty", "non_english", "off_topic", "too_short",
              "scaffolding_only", "dose_quarantined", "banned_abstraction", "generation_tag", "kept",
              "kept_east_africa"):
        if stats[k]:
            pct = f"  ({stats[k] / max(stats['total'], 1):.1%})" if k != "total" else ""
            print(f"  {k:22} {stats[k]:7,}{pct}")

    if not kept:
        print("nothing survived filtering", file=sys.stderr)
        return 1

    # Shard so prepare_sft_data.py's per-file reporting stays readable.
    n = 0
    for i in range(0, len(kept), args.shard_size):
        shard = kept[i:i + args.shard_size]
        p = args.out_dir / f"stage1-{i // args.shard_size:03d}.jsonl"
        p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in shard),
                     encoding="utf-8")
        n += 1
    print(f"\nwrote {n} shard(s) of up to {args.shard_size} rows to {args.out_dir}")

    if quarantined:
        qp = args.out_dir.parent / "_stage1_dose_quarantine.jsonl"
        qp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n"
                              for r in quarantined), encoding="utf-8")
        print(f"quarantined {len(quarantined):,} dose-bearing rows -> {qp}")
        print("  (kept on disk for inspection; deliberately NOT training material)")

    sha = hashlib.sha256(src.read_bytes()).hexdigest()
    manifest = {
        "source_dataset": HF_DATASET,
        "source_file": str(src),
        "source_sha256": sha,
        "source_license": "apache-2.0",
        "filters_applied": {
            "non_english": "Devanagari, or 4+ French function words",
            "off_topic": "no agricultural vocabulary anywhere in the row",
            "too_short": f"answer under {args.min_answer_words} words",
            "dose_quarantined": "verify_corpus.py DOSE_RATE, plus tons/cartloads/bags "
                                "per acre|hectare and NPK ratios",
            "banned_abstraction": "verify_corpus.py BANNED_ABSTRACT",
        },
        "counts": dict(stats),
        "shards": n,
    }
    (args.out_dir / "stage1_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"manifest -> {args.out_dir / 'stage1_manifest.json'}")
    print("\nThis corpus is third-party and unverified for factual accuracy. It is "
          "stage 1 only.\nStage 2 on our own verified rows is what the model ships with.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
