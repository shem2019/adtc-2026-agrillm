#!/usr/bin/env python3
"""Prove the loss mask in prepare_sft_data.encode() is correct.

    python3 train/pipeline/test_masking.py          # mock tokenizer, runs anywhere
    python3 train/pipeline/test_masking.py --real    # against the real Qwen tokenizer

This exists because the loss mask is the single most expensive thing to get
wrong here and the cheapest thing to get wrong silently. If the mask leaks past
the assistant span, the model spends capacity learning to write farmer questions
instead of answers; if it stops short of <|im_end|>, the model never learns to
stop, which is half of why the Round 1 checkpoint ran 90 words past a stated
limit.

The mock reproduces Qwen2.5's actual chat template behaviour:
  * every turn renders as  <|im_start|>{role}\\n{content}<|im_end|>\\n
  * if no system message is supplied, the template injects Qwen's own default
  * add_generation_prompt appends  <|im_start|>assistant\\n
and tokenises in a way that is prefix-consistent, which is the property
encode() relies on. Running with --real swaps in the genuine tokenizer and
checks the same invariants against it.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from prepare_sft_data import IGNORE_INDEX, encode  # noqa: E402

QWEN_DEFAULT_SYSTEM = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."


class MockQwenTokenizer:
    """Faithful-enough stand-in: real template semantics, toy vocabulary."""

    def __init__(self) -> None:
        self.vocab: dict[str, int] = {}

    def _id(self, piece: str) -> int:
        if piece not in self.vocab:
            self.vocab[piece] = len(self.vocab) + 1
        return self.vocab[piece]

    def render(self, messages: list[dict], add_generation_prompt: bool = False) -> str:
        out = []
        if not messages or messages[0]["role"] != "system":
            out.append(f"<|im_start|>system\n{QWEN_DEFAULT_SYSTEM}<|im_end|>\n")
        for m in messages:
            out.append(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n")
        if add_generation_prompt:
            out.append("<|im_start|>assistant\n")
        return "".join(out)

    def _tokenize(self, text: str) -> list[int]:
        # split on special tokens, then on whitespace -- deterministic and
        # prefix-consistent, which is all encode() assumes about a tokenizer
        parts = re.split(r"(<\|im_start\|>|<\|im_end\|>|\n)", text)
        ids = []
        for part in parts:
            if not part:
                continue
            if part in ("<|im_start|>", "<|im_end|>", "\n"):
                ids.append(self._id(part))
            else:
                for word in part.split(" "):
                    if word:
                        ids.append(self._id(word))
        return ids

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False):
        text = self.render(messages, add_generation_prompt)
        return self._tokenize(text) if tokenize else text

    def decode(self, ids: list[int]) -> str:
        inv = {v: k for k, v in self.vocab.items()}
        return " ".join(inv.get(i, "?") for i in ids)


CASES = [
    (
        "single turn, no system (grader condition)",
        [
            {"role": "user", "content": "My maize leaves are yellow. What is wrong?"},
            {"role": "assistant", "content": "That pattern points to nitrogen deficiency."},
        ],
        ["That pattern points to nitrogen deficiency."],
    ),
    (
        "single turn, with AgriLLM system prompt",
        [
            {"role": "system", "content": "You are an offline agricultural advisor."},
            {"role": "user", "content": "How much Imidacloprid should I spray?"},
            {"role": "assistant", "content": "I cannot give you a rate. Check the product label."},
        ],
        ["I cannot give you a rate. Check the product label."],
    ),
    (
        "three-turn conversation (two assistant spans)",
        [
            {"role": "user", "content": "I have fall armyworm in my maize."},
            {"role": "assistant", "content": "How bad does it look right now?"},
            {"role": "user", "content": "Maybe one in five plants."},
            {"role": "assistant", "content": "Drop dry wood ash into the whorl of each plant."},
        ],
        ["How bad does it look right now?", "Drop dry wood ash into the whorl of each plant."],
    ),
    (
        "four-turn with system (longest realistic shape)",
        [
            {"role": "system", "content": "You are an offline farming advisor."},
            {"role": "user", "content": "My chickens are dying."},
            {"role": "assistant", "content": "Are you seeing twisted necks?"},
            {"role": "user", "content": "Yes, and greenish droppings."},
            {"role": "assistant", "content": "That points to Newcastle disease. Separate sick birds now."},
        ],
        ["Are you seeing twisted necks?", "That points to Newcastle disease. Separate sick birds now."],
    ),
]


def check(tok, name: str, messages: list[dict], expect_assistant: list[str]) -> bool:
    out = encode(messages, tok, max_len=4096)
    if out is None:
        print(f"  FAIL  {name}: encode() returned None")
        return False

    input_ids, labels = out
    if len(input_ids) != len(labels):
        print(f"  FAIL  {name}: length mismatch {len(input_ids)} vs {len(labels)}")
        return False

    unmasked = [t for t, l in zip(input_ids, labels) if l != IGNORE_INDEX]
    masked_text = tok.decode(unmasked)

    problems = []

    # 1. every expected assistant content must be present in the unmasked region
    for content in expect_assistant:
        words = [w for w in content.replace("\n", " ").split(" ") if w]
        if not all(w in masked_text for w in words):
            problems.append(f"assistant text missing from loss region: {content!r}")

    # 2. no user or system content may leak into the loss region
    for m in messages:
        if m["role"] in ("user", "system"):
            distinctive = [w for w in m["content"].split(" ") if len(w) > 4]
            leaked = [w for w in distinctive if w in masked_text]
            if leaked:
                problems.append(f"{m['role']} text leaked into loss region: {leaked[:4]}")
    if QWEN_DEFAULT_SYSTEM.split(" ")[3] in masked_text and messages[0]["role"] != "system":
        problems.append("Qwen default system prompt leaked into loss region")

    # 3. the stop token must be trained on, once per assistant turn
    n_imend = masked_text.count("<|im_end|>")
    if n_imend != len(expect_assistant):
        problems.append(f"expected {len(expect_assistant)} <|im_end|> in loss region, got {n_imend}")

    # 4. the role header must NOT be trained on (it is part of the prompt)
    if "<|im_start|>" in masked_text:
        problems.append("<|im_start|> header leaked into loss region")

    pct = 100 * len(unmasked) / len(input_ids)
    if problems:
        print(f"  FAIL  {name}")
        for p in problems:
            print(f"          - {p}")
        print(f"          loss region: {masked_text[:160]!r}")
        return False

    print(f"  ok    {name}  ({len(input_ids)} tok, {len(unmasked)} trained = {pct:.0f}%)")
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", action="store_true",
                    help="use the real Qwen tokenizer (needs transformers installed)")
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    args = ap.parse_args()

    if args.real:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.model)
        print(f"tokenizer: real, {args.model}\n")
    else:
        tok = MockQwenTokenizer()
        print("tokenizer: mock (Qwen2.5 ChatML semantics, toy vocab)\n")

    passed = sum(check(tok, name, msgs, expect) for name, msgs, expect in CASES)
    total = len(CASES)

    print(f"\n{passed}/{total} passed")
    if passed != total:
        print("\nDo NOT train until this is green. A wrong loss mask is invisible in")
        print("the loss curve and obvious in the eval, after you have paid for the GPU.")
        return 1

    # negative control: a deliberately corrupted mask must be caught
    print("\nnegative control (a bad mask must fail these same checks)")
    msgs = CASES[0][1]
    ids = tok.apply_chat_template(msgs, tokenize=True)
    bad_labels = list(ids)  # train on EVERYTHING, including the question
    unmasked = tok.decode([t for t, l in zip(ids, bad_labels) if l != IGNORE_INDEX])
    leaked = "maize" in unmasked
    print(f"  {'ok' if leaked else 'FAIL'}    unmasked-everything control leaks user text: {leaked}")
    return 0 if leaked else 1


if __name__ == "__main__":
    sys.exit(main())
