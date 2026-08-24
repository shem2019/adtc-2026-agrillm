#!/usr/bin/env python3
"""Compare base vs fine-tuned output on the prompts that are actually judged.

    python3 train/compare_prompts.py
    python3 train/compare_prompts.py --adapter-path train/adapters

Training loss tells you the model is fitting the corpus. It tells you nothing
about whether the model can still handle a multi-step advisory question — which
is what the judge panel reads, and half of S_acc.

The specific failure this catches: fine-tuning on ~10,000 terse Q&A pairs can
collapse the base model's ability to produce structured practical advice. The
loss curve looks great while the judged behaviour quietly degrades. The only way
to see it is to read both outputs side by side.

Prompts come from metadata.json so this always tests what you actually submitted,
plus a few probes in the same style as the organisers' hidden prompts.
"""
from __future__ import annotations

import argparse
import json
import textwrap
import time
from pathlib import Path

REPO = Path(__file__).parent.parent

# Same register as the submitted prompts: a scenario, a diagnosis, and a
# constrained plan. If the fine-tune has flattened the model into a one-line
# answering machine, it shows up here first.
PROBES = [
    "A farmer's bean crop shows yellowing between the leaf veins on older leaves "
    "first, while new growth looks normal. Diagnose the most likely cause and say "
    "what you would confirm before recommending an input purchase.",

    "Explain to a farmer with no formal training why rotating maize with beans "
    "improves soil fertility. Keep it practical and under 150 words.",

    "A farmer has 1.5 acres, no irrigation, and the rains are two weeks late. "
    "What should they do differently this season?",
]


def load_submitted_prompts() -> list[str]:
    meta = json.loads((REPO / "metadata.json").read_text())
    return [p["prompt"] for p in meta.get("test_prompts", [])]


def show(title: str, text: str, elapsed: float, width: int = 84) -> None:
    print(f"\n{'─' * width}")
    print(f"  {title}   ({elapsed:.1f}s, {len(text.split())} words)")
    print("─" * width)
    for para in text.strip().split("\n"):
        print(textwrap.fill(para, width=width - 2, initial_indent="  ",
                            subsequent_indent="  ") if para.strip() else "")


def gen(model, tok, prompt: str, max_tokens: int) -> str:
    """Call mlx_lm.generate defensively.

    The signature has moved across mlx-lm releases (max_tokens, temp and sampler
    have all shifted). Failing here would waste the several minutes already spent
    loading two 1.5B models, so try the variants rather than crashing.
    """
    from mlx_lm import generate
    for kwargs in ({"max_tokens": max_tokens}, {"max_tokens": max_tokens, "temp": 0.0}, {}):
        try:
            return generate(model, tok, prompt=prompt, verbose=False, **kwargs)
        except TypeError:
            continue
    return generate(model, tok, prompt)          # last resort, library defaults


def chat_prompt(tok, text: str) -> str:
    """Judged path: judges talk to the model, so apply the chat template."""
    try:
        return tok.apply_chat_template(
            [{"role": "user", "content": text}],
            add_generation_prompt=True, tokenize=False)
    except Exception:                             # noqa: BLE001
        return text


def raw_prompt(text: str) -> str:
    """Scored path: lm-eval feeds bare text with NO chat template. This is the
    exact shape prepare_data.py trains on, and the shape the benchmark uses."""
    return f"Question: {text}\nAnswer:"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--adapter-path", default="train/adapters")
    ap.add_argument("--max-tokens", type=int, default=320)
    ap.add_argument("--raw-tokens", type=int, default=120,
                    help="Token budget for the raw lm-eval-style completion.")
    ap.add_argument("--chat-only", action="store_true",
                    help="Skip the raw-completion path.")
    ap.add_argument("--base-only", action="store_true",
                    help="Record a baseline before any adapter exists.")
    args = ap.parse_args()

    from mlx_lm import generate, load

    prompts = load_submitted_prompts() + PROBES
    print(f"{len(prompts)} prompts: {len(prompts) - len(PROBES)} submitted, "
          f"{len(PROBES)} probes")

    adapter = REPO / args.adapter_path
    have_adapter = adapter.exists() and any(adapter.glob("*.safetensors"))
    if args.base_only:
        have_adapter = False
    elif not have_adapter:
        print(f"note: no adapter at {adapter} — showing base model only")

    print("\nloading base model…")
    base_model, base_tok = load(args.model)

    tuned = None
    if have_adapter:
        print(f"loading fine-tuned model (adapter: {adapter})…")
        tuned = load(args.model, adapter_path=str(adapter))

    stats = {"base_chat": [], "tuned_chat": [], "base_raw": [], "tuned_raw": []}

    for i, prompt in enumerate(prompts, 1):
        kind = "SUBMITTED" if i <= len(prompts) - len(PROBES) else "PROBE"
        print(f"\n\n{'=' * 84}")
        print(f"[{i}/{len(prompts)}] {kind}")
        print(f"{'=' * 84}")
        print(textwrap.fill(prompt, width=82, initial_indent="  ",
                            subsequent_indent="  "))

        # --- judged path: conversational -------------------------------------
        print("\n  ### CHAT MODE (what the judge panel sees)")
        t0 = time.time()
        out = gen(base_model, base_tok, chat_prompt(base_tok, prompt), args.max_tokens)
        show("BASE  [chat]", out, time.time() - t0)
        stats["base_chat"].append(len(out.split()))

        if tuned:
            m, tk = tuned
            t0 = time.time()
            out = gen(m, tk, chat_prompt(tk, prompt), args.max_tokens)
            show("TUNED [chat]", out, time.time() - t0)
            stats["tuned_chat"].append(len(out.split()))

        # --- scored path: raw completion, no chat template --------------------
        if not args.chat_only:
            print("\n  ### RAW MODE (what lm-eval scores — no chat template)")
            t0 = time.time()
            out = gen(base_model, base_tok, raw_prompt(prompt), args.raw_tokens)
            show("BASE  [raw]", out, time.time() - t0)
            stats["base_raw"].append(len(out.split()))

            if tuned:
                m, tk = tuned
                t0 = time.time()
                out = gen(m, tk, raw_prompt(prompt), args.raw_tokens)
                show("TUNED [raw]", out, time.time() - t0)
                stats["tuned_raw"].append(len(out.split()))

    def avg(xs): return sum(xs) / len(xs) if xs else 0.0

    print(f"\n\n{'=' * 84}")
    print("LENGTH SUMMARY  (the objective test for catastrophic forgetting)")
    print("=" * 84)
    print(f"  {'':<14}{'BASE':>10}{'TUNED':>10}{'CHANGE':>12}")
    for label, bk, tk_ in (("chat mode", "base_chat", "tuned_chat"),
                           ("raw mode", "base_raw", "tuned_raw")):
        b, t = avg(stats[bk]), avg(stats[tk_])
        if not stats[tk_]:
            print(f"  {label:<14}{b:>10.0f}{'-':>10}{'(no adapter)':>12}")
            continue
        delta = (t - b) / b * 100 if b else 0.0
        flag = "  <-- COLLAPSE" if delta < -40 else ("  <-- shorter" if delta < -20 else "")
        print(f"  {label:<14}{b:>10.0f}{t:>10.0f}{delta:>11.0f}%{flag}")
    print("""
  Chat mode shrinking more than ~40% means the fine-tune has flattened the
  model into a terse answering machine. That costs you the judged half of
  S_acc. Raw mode getting shorter is fine, even desirable -- that path is
  scored on whether the right continuation is probable, not on length.
""")

    print(f"{'=' * 84}")
    print("""What to look for, in priority order:

  1. Did the fine-tuned answer get SHORTER and less structured? That is the
     catastrophic-forgetting signature. It is the main risk of this whole
     approach and the reason to keep training light.
  2. Did it gain real agronomy specificity — named pests, actual rates, correct
     units — or just change its tone?
  3. Does it still answer every part of a multi-part question, or does it now
     answer only the first clause?
  4. Any LaTeX, "Therefore:", or dataset artefacts leaking into prose? That
     means the training format has bled into the judged register.

If 1 or 3 look bad, retrain with fewer iterations or a lower learning rate.
A model that scores marginally better on the benchmark but writes worse advice
is a bad trade — the judged half is 50% of S_acc.""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
