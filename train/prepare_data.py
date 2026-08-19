#!/usr/bin/env python3
"""Build the agriculture fine-tuning corpus for Qwen2.5-1.5B-Instruct.

Outputs MLX-LM's expected layout:
    train/data/train.jsonl
    train/data/valid.jsonl
    train/data/test.jsonl

FORMAT DECISION — this is the part that matters most.

The ADTC profiler scores accuracy through lm-eval's `loglikelihood` path: it
feeds the model `context + continuation` as RAW TEXT with no chat template, and
picks whichever continuation has the highest log-probability. No system prompt,
no <|im_start|> wrapper, no politeness.

So we train on plain `Question:/Answer:` completions rather than chat-formatted
turns. A model fine-tuned to open with "Certainly! Here's a helpful overview..."
spends probability mass on tokens that earn zero points and slow generation
down — and generation speed is 30% of the score.

Sources are declared in SOURCES below with their licences. The repo ships under
GPL-3.0 and judges read REPORT.md, so provenance is written out to
train/data/SOURCES.md on every run — do not ship a corpus you cannot account for.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

HERE = Path(__file__).parent
OUT = HERE / "data"
AFRICAN_DIR = HERE / "african"

MIN_Q, MAX_Q = 15, 400
MIN_A, MAX_A = 20, 1200

# Openers that waste tokens and teach the model to be verbose. Stripped so the
# answer starts on the substance.
FILLER = re.compile(
    r"^\s*(certainly[!,.]?|sure[!,.]?|of course[!,.]?|absolutely[!,.]?|"
    r"great question[!,.]?|i'd be happy to help[!,.]?|here'?s?( is)? "
    r"(a |an |the )?(helpful |brief |short |detailed )?(overview|answer|"
    r"explanation)[:,.]?)\s*",
    re.IGNORECASE,
)


def clean(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.replace("\r\n", "\n").replace(" ", " ")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    prev = None
    while prev != s:            # openers sometimes stack: "Sure! Certainly, ..."
        prev = s
        s = FILLER.sub("", s).strip()
    return s.strip()


def usable(q: str, a: str) -> bool:
    if not (MIN_Q <= len(q) <= MAX_Q) or not (MIN_A <= len(a) <= MAX_A):
        return False
    if a.lower().startswith(("i don't know", "i cannot", "as an ai")):
        return False
    if q.count("?") > 4 or a.count("http") > 2:
        return False
    # Answers that merely restate the question teach nothing.
    if a.strip().lower() == q.strip().lower():
        return False
    return True


def render(q: str, a: str) -> str:
    """Plain completion format — deliberately not a chat template."""
    return f"Question: {q}\nAnswer: {a}"


# hf_id, licence, cap, note. `cap` bounds how much any single source can
# dominate — blending several independent corpora beats over-fitting one.
SOURCES = [
    ("KisanVaani/agriculture-qa-english-only", "Apache-2.0", 9000,
     "Broad practical agronomy: crops, soil, livestock. South-Asia weighted."),
    ("manifesta/verified-agronomy-17k", "CC0-1.0", 9000,
     "Quantitative agronomy with every formula traced to a published source. "
     "~10% are deliberately unanswerable, teaching the model to say so."),
    ("45acp/agronomy", "MIT", 3000,
     "Embrapa / Instituto Biologico material — Brazilian provenance, useful "
     "counterweight to the South-Asian bias."),
    ("RayNene/adaption-agronomy-qa-pairs", "UNVERIFIED", 6000,
     "Self-described East Africa Agronomy QA. NO LICENCE DECLARED — verify on "
     "the dataset page before shipping, or drop it."),
]

# Column names vary across datasets; probe in priority order.
Q_KEYS = ("question", "prompt", "instruction", "input", "query", "Question")
A_KEYS = ("answers", "answer", "completion", "output", "response", "Answer")


def _first_field(row: dict, keys) -> str:
    for k in keys:
        v = row.get(k)
        if isinstance(v, str) and v.strip():
            return v
    return ""


def load_hf_source(hf_id: str, licence: str, cap: int) -> list[tuple[str, str]]:
    try:
        from datasets import load_dataset
    except ImportError:
        sys.exit("pip install datasets  (inside your venv)")

    print(f"→ {hf_id}  [{licence}]")
    try:
        ds = load_dataset(hf_id, split="train")
    except Exception as e:                      # noqa: BLE001 - any failure is survivable
        print(f"  ! could not load ({type(e).__name__}: {str(e)[:90]}) — skipping")
        return []

    pairs, skipped = [], 0
    for row in ds:
        q, a = clean(_first_field(row, Q_KEYS)), clean(_first_field(row, A_KEYS))
        if usable(q, a):
            pairs.append((q, a))
        else:
            skipped += 1

    if not pairs:
        print(f"  ! no usable pairs — columns were {list(ds.features)[:6]}")
        return []

    random.shuffle(pairs)
    pairs = pairs[:cap]
    print(f"  {len(ds):,} rows -> {len(pairs):,} kept ({skipped:,} filtered out)")
    return pairs


def load_local() -> list[tuple[str, str]]:
    """Optional hand-curated material: JSONL with {"question","answer"} per line."""
    if not AFRICAN_DIR.exists():
        return []
    pairs = []
    for path in sorted(AFRICAN_DIR.glob("*.jsonl")):
        n = 0
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                print(f"  ! malformed line in {path.name}")
                continue
            q, a = clean(_first_field(row, Q_KEYS)), clean(_first_field(row, A_KEYS))
            if usable(q, a):
                pairs.append((q, a))
                n += 1
        print(f"  {path.name}: {n} pairs")
    return pairs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-unverified", action="store_true",
                    help="Exclude any source whose licence is not confirmed.")
    ap.add_argument("--local-weight", type=int, default=3,
                    help="Repeat each hand-curated pair this many times so a "
                         "small local set is not drowned out.")
    args = ap.parse_args()

    random.seed(args.seed)
    OUT.mkdir(parents=True, exist_ok=True)

    collected, manifest = [], []
    for hf_id, licence, cap, note in SOURCES:
        if args.skip_unverified and licence == "UNVERIFIED":
            print(f"→ {hf_id} — SKIPPED (--skip-unverified)")
            manifest.append((hf_id, licence, 0, note + " [skipped]"))
            continue
        pairs = load_hf_source(hf_id, licence, cap)
        collected += pairs
        manifest.append((hf_id, licence, len(pairs), note))

    print("→ local curated material")
    local = load_local()
    if not local:
        print(f"  none in {AFRICAN_DIR.name}/ (optional)")

    # Dedupe on the question; local entries take precedence over scraped ones.
    seen, merged = set(), []
    for q, a in local * args.local_weight + collected:
        k = re.sub(r"[^a-z0-9 ]", "", q.lower()).strip()
        if k in seen:
            continue
        seen.add(k)
        merged.append((q, a))

    dupes = len(local) * args.local_weight + len(collected) - len(merged)
    print(f"\n→ merged {len(merged):,} unique ({dupes:,} cross-source duplicates removed)")

    random.shuffle(merged)
    n = len(merged)
    n_valid = max(50, int(n * 0.05))
    n_test = max(50, int(n * 0.05))
    splits = {
        "valid": merged[:n_valid],
        "test": merged[n_valid:n_valid + n_test],
        "train": merged[n_valid + n_test:],
    }

    for name, rows in splits.items():
        path = OUT / f"{name}.jsonl"
        with path.open("w", encoding="utf-8") as fh:
            for q, a in rows:
                fh.write(json.dumps({"text": render(q, a)}, ensure_ascii=False) + "\n")
        chars = sum(len(render(q, a)) for q, a in rows)
        print(f"  {path.relative_to(HERE.parent)}: {len(rows):>6} examples  "
              f"~{chars // 4:,} tokens")

    # Provenance record. REPORT.md needs this, and an unaccountable corpus is
    # the kind of thing that unravels under judge scrutiny.
    lines = ["# Training corpus provenance", "",
             f"Generated by `train/prepare_data.py` (seed {args.seed}).", "",
             "| Source | Licence | Pairs used | Notes |",
             "|---|---|---:|---|"]
    for hf_id, licence, cnt, note in manifest:
        lines.append(f"| [{hf_id}](https://huggingface.co/datasets/{hf_id}) "
                     f"| {licence} | {cnt:,} | {note} |")
    if local:
        lines.append(f"| `train/african/*.jsonl` (hand-curated) | own work "
                     f"| {len(local):,} (x{args.local_weight}) | Local material. |")
    lines += ["", f"**Total unique examples: {n:,}** "
                  f"(train {len(splits['train']):,} / valid {len(splits['valid']):,} "
                  f"/ test {len(splits['test']):,})", ""]
    unverified = [m for m in manifest if m[1] == "UNVERIFIED" and m[2] > 0]
    if unverified:
        lines += ["> ⚠ **Unverified licences included.** Confirm on the dataset "
                  "page or re-run with `--skip-unverified` before submitting.", ""]
    (OUT / "SOURCES.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"  {(OUT / 'SOURCES.md').relative_to(HERE.parent)}: provenance table")

    print(f"\n✓ {n:,} unique examples from {sum(1 for m in manifest if m[2])} sources")
    if unverified:
        print("⚠ Includes sources with no declared licence:")
        for hf_id, _, cnt, _ in unverified:
            print(f"    {hf_id} ({cnt:,} pairs)")
        print("  Verify on huggingface.co or re-run with --skip-unverified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
