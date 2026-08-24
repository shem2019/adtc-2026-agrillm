#!/usr/bin/env python3
"""Build the agriculture fine-tuning corpus for Qwen2.5-1.5B-Instruct.

Outputs MLX-LM's expected layout:
    train/data/train.jsonl
    train/data/valid.jsonl
    train/data/test.jsonl

FORMAT DECISION — and a correction.

We first trained on raw `Question:/Answer:` completions, reasoning that the
profiler scores accuracy through lm-eval's `loglikelihood` path, which feeds
`context + continuation` as bare text with no chat template.

That was the wrong target. The official ADTC FAQ states that S_acc is graded by
a judge panel that "chats with it live through our in-browser interface" — the
lm-eval task in the profiler is a self-check, not the score. Training every one
of 16,000 examples as a single-turn completion taught the model a Q&A reflex and
measurably degraded its conversational ability: it looped, and it answered only
the first clause of multi-part questions.

Default is now `--format chat`, which wraps each pair in the model's own chat
template so training matches evaluation. `--format completion` keeps the old
behaviour for comparison.

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
# Deliberately the _clean/ subdirectory, not african/ itself: training must read
# only what validate_african.py has passed. Pointing at the raw batches would let
# code fences, filler openers and duplicates straight into the corpus.
AFRICAN_DIR = HERE / "african" / "_clean"

MIN_Q, MAX_Q = 15, 500
# MIN_A was 20 and that was the real bug. Quantitative agronomy answers are
# SHORT -- "5.87 kg/ha" is 10 characters and is a complete, correct answer.
# A 20-character floor silently deleted 11,095 of manifesta's 17,199 rows while
# keeping its "cannot be determined without X" refusals (~45 chars), inverting
# the corpus: 27% of examples became punts. Length is a terrible proxy for
# usefulness on numeric data.
MIN_A, MAX_A = 3, 3000

# Refusals belong in the corpus -- an advisor that invents a fertiliser rate is
# worse than one that asks for a soil test. But they must stay a minority
# behaviour, not the default.
MAX_REFUSAL_SHARE = 0.08

REFUSAL = re.compile(
    r"(cannot be (determined|answered|calculated)|insufficient (data|"
    r"information)|not enough information|unable to determine|"
    r"requires? (more|additional) (data|information))",
    re.IGNORECASE,
)

# A short answer that is a number with an optional unit is legitimate and must
# bypass the English-prose check, which needs >=4 words to fire.
NUMERIC_ANSWER = re.compile(
    r"^\s*[~<>=]?\s*-?[\d,]+(\.\d+)?\s*"
    r"(%|[a-zA-Z°µ/·\^\-\d\s\.]{0,24})?\s*$"
)

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
    # Strip MATH-benchmark LaTeX. manifesta closes every worked solution with
    # "The final answer is $\\boxed{4.64 t/ha}$". An exact-match grader looking
    # for "4.64 t/ha" fails against the boxed form, and a farmer reading an
    # advisory should never be shown LaTeX. Unwrap to the plain value.
    s = re.sub(r"\$?\\boxed\{([^{}]*)\}\$?", r"\1", s)
    s = re.sub(r"\n*\s*The final answer is\s*", "\nTherefore: ", s)
    s = s.replace("\\%", "%").replace("\\times", "x").replace("\\,", " ")
    s = re.sub(r"\$([^$\n]{1,40})\$", r"\1", s)
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    prev = None
    while prev != s:            # openers sometimes stack: "Sure! Certainly, ..."
        prev = s
        s = FILLER.sub("", s).strip()
    return s.strip()


def usable(q: str, a: str) -> tuple[bool, str]:
    """Returns (keep, reason_if_dropped) so the run can report WHY rows were cut."""
    if not (MIN_Q <= len(q) <= MAX_Q) or not (MIN_A <= len(a) <= MAX_A):
        return False, "length"
    if a.lower().startswith(("i don't know", "i cannot", "as an ai")):
        return False, "refusal"
    if q.count("?") > 4 or a.count("http") > 2:
        return False, "junk"
    if a.strip().lower() == q.strip().lower():
        return False, "restates question"
    q_ok = is_english(q) or is_swahili(q)
    a_ok = is_english(a) or is_swahili(a) or bool(NUMERIC_ANSWER.match(a))
    if not q_ok or not a_ok:
        return False, "not english/swahili"
    if OFF_DOMAIN.search(q) or OFF_DOMAIN.search(a):
        return False, "off-domain drift"
    blob = q + " " + a
    if not (AGRI_TERMS.search(blob) or SW_AGRI.search(blob)):
        return False, "no agriculture content"
    return True, ""


def render(q: str, a: str) -> str:
    """Raw completion format.

    Originally the default, chosen because lm-eval scores `context +
    continuation` as bare text. That turned out to be the wrong target: the
    official FAQ states S_acc is graded by a judge panel chatting with the model
    live in a browser. Training every example as "Question:/Answer:" taught a
    single-turn reflex and cost the model its conversational ability.

    Kept as a fallback via --format completion.
    """
    return f"Question: {q}\nAnswer: {a}"


def render_chat(q: str, a: str) -> dict:
    """Chat format — matches how the model is actually evaluated.

    mlx-lm applies the model's own chat template to `messages`, so the model
    learns the same <|im_start|>user / <|im_start|>assistant structure it will
    see from the judges. No system message: the judges will not send one, and
    training with a system prompt that is absent at inference creates a
    mismatch the model has to guess its way out of.
    """
    return {"messages": [{"role": "user", "content": q},
                         {"role": "assistant", "content": a}]}


# hf_id, licence, cap, note. `cap` bounds how much any single source can
# dominate — blending several independent corpora beats over-fitting one.
SOURCES = [
    ("KisanVaani/agriculture-qa-english-only", "Apache-2.0", 9000,
     "Broad practical agronomy: crops, soil, livestock. South-Asia weighted."),
    ("manifesta/verified-agronomy-17k", "CC0-1.0", 6000,
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
# `enhanced_completion` first: where a dataset ships both a terse original and a
# rewritten long answer, the rewrite is better written — but see OFF_DOMAIN,
# because rewrites also drift off-topic more often.
Q_KEYS = ("question", "prompt", "instruction", "input", "query", "Question")
A_KEYS = ("enhanced_completion", "answers", "answer", "completion",
          "output", "response", "Answer")

# --- domain and language gates -------------------------------------------------
# Applied to every source. Two independent failure modes seen in real data:
#
#  1. Homonym drift. A row asking about CMD (Cassava Mosaic Disease) answered
#     with "CMD ezali programme ya Windows". Calcium-in-plants answered with
#     calcium in the human body. Sexing insects answered with human gender
#     identity. These are confidently wrong and would teach the model nonsense.
#
#  2. Language dilution. The scored benchmark is English multiple-choice. A 1.5B
#     model has limited capacity; spending it on languages the evaluation never
#     tests risks eroding the English reasoning that actually earns S_acc.
#     Filtering to English is a scoring decision, not a judgement about the
#     languages — multilingual coverage is valuable, just not for this scoreboard.

OFF_DOMAIN = re.compile(
    r"\b(command prompt|windows|microsoft|operating system|"
    r"human body|pregnan\w+|miscarriage|gender identity|"
    r"biological sex|menstrua\w+|contracepti\w+|"
    r"i am an ai|as an ai|nazali ai|i am not a doctor)\b",
    re.IGNORECASE,
)

AGRI_TERMS = re.compile(
    r"\b(crop|farm\w*|soil|seed\w*|plant\w*|harvest\w*|fertili[sz]\w*|"
    r"irrigat\w*|pest\w*|weed\w*|maize|cassava|bean\w*|rice|wheat|sorghum|"
    r"millet|coffee|tea|banana|potato\w*|tomato\w*|livestock|cattle|goat\w*|"
    r"poultry|manure|compost|agronom\w*|agricultur\w*|yield|germinat\w*|"
    r"nitrogen|phosphor\w*|potassium|nutrient\w*|acre|hectare|tillage|"
    r"mulch\w*|rotation|pesticide|herbicide|fungicide|drought|rainfall|"
    r"cultivar|variet\w+|disease|insect|larva\w*|aphid\w*|whitefl\w+|"
    r"striga|weevil|blight|rot|wilt|spacing|planting|pruning|grazing|"
    r"kg/ha|t/ha|plants?/ha|seeding|sowing|evapotranspiration|loam|clay|"
    r"silt|pasture|forage|silage|orchard|greenhouse|hydroponic\w*)s?\b",
    re.IGNORECASE,
)
# The trailing `s?` matters more than it looks: without it `\bacre\b` fails to
# match "acres", and "How many acres is 336.4 hectares?" was being discarded as
# having no agricultural content. Plurals are the common case in real questions.

# Frequent English function words — a cheap, dependency-free language check.
EN_MARKERS = re.compile(
    r"\b(the|and|is|are|of|to|in|for|with|that|this|it|be|as|on|you|can|"
    r"what|how|why|when|which|should|will|from|by|at|or|not|have|has)\b",
    re.IGNORECASE,
)


# Swahili function words. Same trick as EN_MARKERS: cheap, dependency-free, and
# conservative. Needed because the corpus is deliberately English-first for the
# benchmark, but the African Language bonus (+15% on the panel score) requires
# demonstrable Swahili capability -- so Swahili rows must survive the filter that
# exists to keep every OTHER language out.
SW_MARKERS = re.compile(
    r"\b(na|ya|wa|kwa|ni|katika|hii|hizo|kama|lakini|au|kwenye|cha|vya|za|la|"
    r"yake|zao|kuwa|ili|baada|kabla|zaidi|pia|hata|kila|wakati|ambayo|hiyo|"
    r"kwamba|yako|yangu|unaweza|inaweza|hufanya|husaidia)\b",
    re.IGNORECASE,
)

SW_AGRI = re.compile(
    r"\b(kilimo|mkulima|wakulima|shamba|mashamba|udongo|mbegu|mbolea|samadi|"
    r"wadudu|magonjwa|ugonjwa|kupanda|kupalilia|mavuno|kuvuna|umwagiliaji|"
    r"mahindi|maharagwe|mihogo|muhogo|viazi|mtama|ulezi|ndizi|kahawa|chai|"
    r"nyanya|vitunguu|ng.ombe|mbuzi|kondoo|kuku|mifugo|viwavijeshi|kutu|"
    r"ukungu|mvua|msimu|mimea|mmea|majani|mizizi|matunda|virutubisho|"
    r"kunyunyiza|dawa|ugani|mazao|zao|kilele|shina)\b",
    re.IGNORECASE,
)


def is_swahili(text: str) -> bool:
    words = re.findall(r"[a-zA-Z']+", text)
    if len(words) < 4:
        return False
    return len(SW_MARKERS.findall(text)) / len(words) > 0.12


def is_english(text: str) -> bool:
    """Heuristic, not a language ID model — but it has no dependencies and the
    failure mode is conservative (drops borderline rows rather than keeping
    them)."""
    words = re.findall(r"[a-zA-Z']+", text)
    if len(words) < 4:
        return False
    hits = len(EN_MARKERS.findall(text))
    non_ascii = sum(1 for c in text if ord(c) > 127)
    return hits / len(words) > 0.12 and non_ascii / max(len(text), 1) < 0.02


def _first_field(row: dict, keys) -> str:
    for k in keys:
        v = row.get(k)
        if isinstance(v, str) and v.strip():
            return v
    return ""


def build_answer(row: dict) -> str:
    """Assemble the answer, using a worked solution where the dataset has one.

    manifesta splits its content across two columns: `worked_solution` holds the
    derivation, `answer` holds only the final figure. Training on `answer` alone
    teaches the model to emit a number with no reasoning; training on
    `worked_solution` alone leaves the final figure implicit. Joining them
    teaches the derivation AND the answer format the benchmark scores on.
    """
    final = clean(_first_field(row, A_KEYS))
    worked = clean(row.get("worked_solution") or "")
    if not worked:
        return final
    if not final:
        return worked
    if final.lower() in worked.lower()[-len(final) - 40:]:
        return worked                       # already ends with the figure
    unit = (row.get("answer_unit") or "").strip()
    tail = f"{final} {unit}".strip() if unit and unit not in final else final
    return f"{worked}\n\nTherefore: {tail}"


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

    pairs, skipped, reasons = [], 0, {}
    for row in ds:
        q, a = clean(_first_field(row, Q_KEYS)), build_answer(row)
        keep, why = usable(q, a)
        if keep:
            pairs.append((q, a))
        else:
            skipped += 1
            reasons[why] = reasons.get(why, 0) + 1

    if not pairs:
        print(f"  ! no usable pairs — columns were {list(ds.features)[:6]}")
        return []

    random.shuffle(pairs)
    pairs = pairs[:cap]
    top = sorted(reasons.items(), key=lambda kv: -kv[1])[:3]
    detail = ", ".join(f"{k} {v:,}" for k, v in top)
    print(f"  {len(ds):,} rows -> {len(pairs):,} kept ({skipped:,} dropped: {detail})")
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
            q, a = clean(_first_field(row, Q_KEYS)), build_answer(row)
            if usable(q, a)[0]:
                pairs.append((q, a))
                n += 1
        print(f"  {path.name}: {n} pairs")
    return pairs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--format", choices=("chat", "completion"), default="chat",
                    help="chat (default) matches live judge evaluation; "
                         "completion is the older raw Question:/Answer: form.")
    ap.add_argument("--max-seq-length", type=int, default=1024,
                    help="Only used to WARN about truncation. Must match the "
                         "value passed to mlx_lm.lora.")
    ap.add_argument("--skip-unverified", action="store_true",
                    help="Exclude any source whose licence is not confirmed.")
    ap.add_argument("--local-weight", type=int, default=0,
                    help="Repeat each local pair N times. 0 = auto (default): "
                         "computed so local material lands near --local-share.")
    ap.add_argument("--local-share", type=float, default=0.40,
                    help="Target proportion of the corpus from train/african/. "
                         "Only used when --local-weight is auto.")
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

    # Auto-weight. The 3x default existed to stop a small hand-curated set from
    # being drowned by 18,000 scraped rows. Applied blindly to a 10,000-entry
    # generated corpus it does the OPPOSITE — 31,692 rows that swamp everything
    # else. Weight is a function of how much local data there is, so it must be
    # computed, not remembered as a flag.
    weight = args.local_weight
    if weight <= 0:
        if not local:
            weight = 1
        else:
            share = min(max(args.local_share, 0.05), 0.9)
            wanted = share * len(collected) / (1 - share)
            weight = max(1, min(3, round(wanted / len(local))))
        if local:
            projected = len(local) * weight
            pct = projected / (projected + len(collected))
            print(f"  auto local-weight = {weight}x  "
                  f"({len(local):,} local x{weight} vs {len(collected):,} external "
                  f"-> ~{pct:.0%} of corpus, target {args.local_share:.0%})")
    args.local_weight = weight

    # Dedupe on the question; local entries take precedence over scraped ones.
    seen, merged = set(), []
    for q, a in local * args.local_weight + collected:
        k = re.sub(r"[^a-z0-9 ]", "", q.lower()).strip()
        if k in seen:
            continue
        seen.add(k)
        merged.append((q, a))

    # Cap refusals. They are valuable (an advisor that invents a fertiliser rate
    # is worse than one that asks for a soil test) but must not become the
    # model's default response.
    # A refusal is a refusal whether it opens the answer or ends a worked
    # solution. Start-anchored matching missed 1,601 of them last run.
    def _is_refusal(a: str) -> bool:
        # Real refusals in this corpus are ~380 chars and put the verdict LAST:
        #   "Run time follows from ETc... ETo is not provided here...
        #    Therefore: cannot be determined without the reference ET."
        # Start-anchored matching missed all of them, and a 300-char ceiling
        # missed them too. Judge the CONCLUSION, not the opening.
        if REFUSAL.match(a):
            return True
        lines = [ln for ln in a.strip().split("\n") if ln.strip()]
        if lines and REFUSAL.search(lines[-1]):
            return True
        return bool(REFUSAL.search(a[-200:]))

    refusals = [(q, a) for q, a in merged if _is_refusal(a)]
    answers_ = [(q, a) for q, a in merged if not _is_refusal(a)]
    allowed = int(len(answers_) * MAX_REFUSAL_SHARE / (1 - MAX_REFUSAL_SHARE))
    if len(refusals) > allowed:
        print(f"  refusals: {len(refusals):,} -> {allowed:,} "
              f"(capped at {MAX_REFUSAL_SHARE:.0%} of corpus)")
        random.shuffle(refusals)
        refusals = refusals[:allowed]
    merged = answers_ + refusals

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

    # ~4 chars per token for English, plus ~25 tokens of chat-template scaffolding.
    def est_tokens(q, a):
        return (len(q) + len(a)) // 4 + (25 if args.format == "chat" else 6)

    over = 0
    for name, rows in splits.items():
        path = OUT / f"{name}.jsonl"
        with path.open("w", encoding="utf-8") as fh:
            for q, a in rows:
                rec = render_chat(q, a) if args.format == "chat" else {"text": render(q, a)}
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        toks = sum(est_tokens(q, a) for q, a in rows)
        over += sum(1 for q, a in rows if est_tokens(q, a) > args.max_seq_length)
        print(f"  {path.relative_to(HERE.parent)}: {len(rows):>6} examples  "
              f"~{toks:,} tokens")

    print(f"\n  format: {args.format}")
    pct = over / max(n, 1)
    print(f"  examples exceeding --max-seq-length {args.max_seq_length}: "
          f"{over:,} ({pct:.1%})")
    if pct > 0.02:
        # Truncated examples end mid-sentence and teach the model to stop
        # abruptly. Better to know before burning a training run.
        print("    ^ These will be TRUNCATED mid-answer during training, which "
              "teaches\n      the model to stop mid-sentence. Raise "
              "--max-seq-length on mlx_lm.lora\n      to 1536, or lower MAX_A.")

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
