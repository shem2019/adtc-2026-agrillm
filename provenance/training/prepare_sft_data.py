#!/usr/bin/env python3
"""Turn train/african/_clean/*.jsonl into tokenised SFT tensors for full fine-tuning.

    python3 train/pipeline/prepare_sft_data.py \
        --corpus-dir train/african/_clean \
        --out-dir    train/pipeline/data \
        --base-model Qwen/Qwen2.5-1.5B-Instruct

Three things this script exists to get right, because each of them broke a
previous run:

1. CHAT FORMAT, NOT RAW COMPLETION.
   The Round 1 checkpoint was trained on raw "Q:\nA:" text and then served
   through a ChatML template at inference. The model had never seen
   <|im_start|>assistant in training, so it drifted, rambled past the stop
   token, and blew stated word limits. Everything here goes through
   tokenizer.apply_chat_template, which is the exact string llama.cpp will
   build at inference from the GGUF's embedded template.

2. LOSS ON THE ANSWER ONLY.
   If you compute loss over the question tokens too, the model spends capacity
   learning to generate plausible *farmer questions*. We mask everything except
   assistant spans (content + <|im_end|>). Training on <|im_end|> is what
   teaches it to stop, which is half of obeying "under 150 words".

3. SYSTEM-PROMPT MIX THAT MATCHES THE GRADER.
   Qwen2.5's chat template silently injects
       "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
   whenever no system message is supplied. The ADTC graders call the model with
   stock llama.cpp defaults and no system prompt, so that Qwen default string is
   what the model will actually see when it is scored. If we trained only with
   our own AgriLLM system prompt, scoring conditions would be off-distribution.
   So we train on a deliberate mix (see SYSTEM_MIX below): the safety and
   register behaviour has to be intrinsic to the weights, not conditional on a
   system prompt the grader will never send.

Outputs into --out-dir:
    train.pt / val.pt        tokenised tensors (input_ids, labels)
    manifest.json            per-file row counts, SHA256 of every source file,
                             tokenizer + base-model identity, mix ratios
    sample_rendered.txt      20 fully-rendered examples with the masked region
                             marked, so a human (or a judge) can see exactly
                             what the model was trained on
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

# torch and transformers are imported inside main() rather than at module level
# so that the pure logic below -- in particular encode(), which carries the loss
# mask and is the most dangerous function in this file -- can be imported and
# unit-tested anywhere, including on a machine with neither installed.
# See test_masking.py.

# --- system prompt mix ------------------------------------------------------
# Strings live in prompts.py so that the training mix and the default baked into
# the GGUF by set_chat_template_default.py cannot drift apart. See that module
# for why the mix is shaped this way.
sys.path.insert(0, str(Path(__file__).parent))
from prompts import (  # noqa: E402
    AGRI_DEFAULT,
    AGRI_PARAPHRASES,
    NEUTRAL,
    SYSTEM_MIX,
)

IGNORE_INDEX = -100


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_rows(corpus_dir: Path) -> tuple[list[dict], dict]:
    """Read every *.jsonl. Return (rows, per_file_stats).

    Handles both shapes present in the corpus:
        {"question": ..., "answer": ...}          23 files
        {"messages": [{"role","content"}, ...]}   multiturn.jsonl
    """
    rows: list[dict] = []
    stats: dict[str, dict] = {}

    for path in sorted(corpus_dir.glob("*.jsonl")):
        kept = skipped = 0
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"  ! {path.name}:{lineno} bad JSON, skipped ({exc})", file=sys.stderr)
                skipped += 1
                continue

            if "messages" in obj:
                msgs = [
                    {"role": m["role"], "content": str(m["content"]).strip()}
                    for m in obj["messages"]
                    if m.get("role") in ("user", "assistant") and str(m.get("content", "")).strip()
                ]
            else:
                q = str(obj.get("question", "")).strip()
                a = str(obj.get("answer", "")).strip()
                if not q or not a:
                    skipped += 1
                    continue
                msgs = [
                    {"role": "user", "content": q},
                    {"role": "assistant", "content": a},
                ]

            # must be a well-formed alternating conversation ending on assistant
            if len(msgs) < 2 or msgs[0]["role"] != "user" or msgs[-1]["role"] != "assistant":
                skipped += 1
                continue

            rows.append({"messages": msgs, "source": path.name})
            kept += 1

        stats[path.name] = {"kept": kept, "skipped": skipped, "sha256": sha256(path)}
        print(f"  {path.name:34s} {kept:5d} rows" + (f"  ({skipped} skipped)" if skipped else ""))

    return rows, stats


def pick_system(rng: random.Random) -> tuple[str, str | None]:
    """Draw a (label, system_prompt) pair from the mix.

    A None prompt means: pass no system message at all, so the chat template
    injects its own default. Once set_chat_template_default.py has run, that
    default is AGRI_DEFAULT -- i.e. this share trains the model under the exact
    string a grader's prompt-less call will produce.
    """
    r = rng.random()
    acc = 0.0
    for label, prompt, weight in SYSTEM_MIX:
        acc += weight
        if r <= acc:
            if prompt == "__PARAPHRASE__":
                return label, rng.choice(AGRI_PARAPHRASES)
            if prompt == "__NEUTRAL__":
                return label, rng.choice(NEUTRAL)
            return label, prompt
    label, prompt, _ = SYSTEM_MIX[-1]
    return label, (rng.choice(NEUTRAL) if prompt == "__NEUTRAL__" else prompt)


def template_ids(tok, messages: list[dict], add_generation_prompt: bool) -> list[int]:
    """apply_chat_template -> a plain list of token ids, across transformers versions.

    Newer transformers returns a BatchEncoding from apply_chat_template(tokenize=True)
    because return_dict defaults to True; older versions return list[int]. len() on a
    BatchEncoding is the number of KEYS - 2 - so prefix-consistency checks silently
    compare 2 against 2 and reject every row. That is exactly what happened here:
    74,696 of 74,696 rows dropped as "mask misalignment" on a box whose transformers
    was newer than the one the code was written against.

    Normalise rather than pin a version: return_dict=False is not accepted by older
    builds, so handle whatever comes back.
    """
    out = tok.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=add_generation_prompt
    )
    if hasattr(out, "input_ids"):          # BatchEncoding / dict-like
        out = out["input_ids"]
    if len(out) and isinstance(out[0], (list, tuple)):   # batched
        out = out[0]
    ids = list(out)
    if ids and not isinstance(ids[0], int):
        raise TypeError(
            f"apply_chat_template gave {type(ids[0]).__name__} tokens, expected int. "
            f"transformers API changed again; fix template_ids()."
        )
    return ids


def encode(messages: list[dict], tok, max_len: int) -> tuple[list[int], list[int]] | None:
    """Render through the chat template; unmask assistant spans only.

    For ChatML the rendering is strictly prefix-consistent, so the token span of
    assistant turn i is:
        render(messages[:i], add_generation_prompt=True)   -> prefix
        render(messages[:i+1])                             -> prefix + span
    We verify that prefix-consistency per row and drop the row if it ever fails,
    rather than silently training on a misaligned mask.
    """
    full_ids = template_ids(tok, messages, add_generation_prompt=False)
    labels = [IGNORE_INDEX] * len(full_ids)

    for i, msg in enumerate(messages):
        if msg["role"] != "assistant":
            continue
        prefix_ids = template_ids(tok, messages[:i], add_generation_prompt=True)
        upto_ids = template_ids(tok, messages[: i + 1], add_generation_prompt=False)

        # prefix-consistency checks
        if len(prefix_ids) >= len(upto_ids) or prefix_ids != upto_ids[: len(prefix_ids)]:
            return None
        if upto_ids != full_ids[: len(upto_ids)]:
            return None

        for j in range(len(prefix_ids), len(upto_ids)):
            labels[j] = full_ids[j]

    if all(l == IGNORE_INDEX for l in labels):
        return None
    if len(full_ids) > max_len:
        return None  # corpus p99 is ~560 tokens; anything over max_len is anomalous
    return full_ids, labels


def main() -> int:
    import torch
    from transformers import AutoTokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus-dir", type=Path, default=Path("train/african/_clean"))
    ap.add_argument("--out-dir", type=Path, default=Path("train/pipeline/data"))
    ap.add_argument("--base-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--revision", default=None, help="pin the exact base-model commit SHA")
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--val-frac", type=float, default=0.03)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    if not args.corpus_dir.is_dir():
        print(f"corpus dir not found: {args.corpus_dir}", file=sys.stderr)
        return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    print(f"reading corpus from {args.corpus_dir}")
    rows, file_stats = load_rows(args.corpus_dir)
    print(f"  -> {len(rows)} conversations from {len(file_stats)} files")

    print(f"loading tokenizer: {args.base_model}" + (f" @ {args.revision}" if args.revision else ""))
    tok = AutoTokenizer.from_pretrained(args.base_model, revision=args.revision)

    encoded: list[dict] = []
    dropped_align = dropped_long = 0
    sys_counts: dict[str, int] = {}

    for row in rows:
        label, sys_prompt = pick_system(rng)
        sys_counts[label] = sys_counts.get(label, 0) + 1
        if sys_prompt is None:
            msgs = row["messages"]
        else:
            msgs = [{"role": "system", "content": sys_prompt}] + row["messages"]

        out = encode(msgs, tok, args.max_len)
        if out is None:
            n_tok = len(template_ids(tok, msgs, add_generation_prompt=False))
            if n_tok > args.max_len:
                dropped_long += 1
            else:
                dropped_align += 1
            continue

        input_ids, labels = out
        encoded.append(
            {
                "input_ids": input_ids,
                "labels": labels,
                "source": row["source"],
                "n_tokens": len(input_ids),
                "system": label,
            }
        )

    if dropped_align:
        print(f"  ! dropped {dropped_align} rows: chat-template mask misalignment")
    if dropped_long:
        print(f"  ! dropped {dropped_long} rows: longer than --max-len {args.max_len}")
    print(f"  -> {len(encoded)} encoded examples")

    lens = sorted(e["n_tokens"] for e in encoded)
    pct = lambda p: lens[max(0, int(len(lens) * p) - 1)]
    print(f"  token length: p50={pct(.50)} p90={pct(.90)} p99={pct(.99)} max={lens[-1]}")

    rng.shuffle(encoded)
    n_val = max(1, int(len(encoded) * args.val_frac))
    val, train = encoded[:n_val], encoded[n_val:]
    print(f"  split: {len(train)} train / {len(val)} val")

    torch.save(train, args.out_dir / "train.pt")
    torch.save(val, args.out_dir / "val.pt")

    # a human-readable window into what the model actually sees
    sample_path = args.out_dir / "sample_rendered.txt"
    with sample_path.open("w", encoding="utf-8") as fh:
        for ex in train[:20]:
            text = tok.decode(ex["input_ids"])
            trained = tok.decode([t for t, l in zip(ex["input_ids"], ex["labels"]) if l != IGNORE_INDEX])
            fh.write("=" * 78 + f"\nsource: {ex['source']}  tokens: {ex['n_tokens']}\n")
            fh.write("--- full rendered sequence ---\n" + text + "\n")
            fh.write("--- tokens loss is computed on ---\n" + trained + "\n\n")

    manifest = {
        "base_model": args.base_model,
        "base_model_revision": args.revision,
        "tokenizer_class": tok.__class__.__name__,
        "max_len": args.max_len,
        "seed": args.seed,
        "counts": {
            "conversations_read": len(rows),
            "encoded": len(encoded),
            "train": len(train),
            "val": len(val),
            "dropped_misaligned": dropped_align,
            "dropped_too_long": dropped_long,
        },
        "token_length": {"p50": pct(.50), "p90": pct(.90), "p99": pct(.99), "max": lens[-1]},
        "system_prompt_mix": sys_counts,
        # Recorded so a reviewer can confirm the string trained under is the same
        # one set_chat_template_default.py bakes into the shipped GGUF.
        "agri_default_system_prompt": AGRI_DEFAULT,
        "source_files": file_stats,
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"\nwrote {args.out_dir}/train.pt, val.pt, manifest.json, sample_rendered.txt")
    print("Read sample_rendered.txt before you launch a run. If the '--- tokens loss")
    print("is computed on ---' block contains anything other than assistant text,")
    print("stop and fix it: that is the bug that cost Round 1.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
