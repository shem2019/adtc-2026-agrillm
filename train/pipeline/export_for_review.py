#!/usr/bin/env python3
"""Dump prompts and the model's raw answers into one readable file for human review.

    # from generations already stored by the eval (instant, no GPU):
    python3 train/pipeline/export_for_review.py \
        train/pipeline/runs/fullft-lr2e-5/eval/checkpoint-816.json \
        -o ~/review.md

    # or ask a running server fresh questions:
    python3 train/pipeline/export_for_review.py --live --prompts my_prompts.txt \
        --url http://127.0.0.1:8080 -o ~/review.md

Why this exists: the string-rule harness graded three of five safety prompts as
total failures when the model had actually answered correctly - it said "health
facility" where the rule wanted "hospital", and "can't" where the rule wanted
"cannot". Rules are fast and reproducible and cannot read. This puts the raw
text in front of a reader.

The eval already stores every sample it drew, so the default mode costs nothing
and reviews exactly the generations that produced the scores.
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path(__file__).parent.parent.parent


def ask(url: str, prompt: str, max_tokens: int = 700) -> str:
    """Stock llama.cpp defaults - the settings the graders actually use."""
    body = json.dumps({
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.8, "repeat_penalty": 1.0, "max_tokens": max_tokens,
    }).encode()
    req = urllib.request.Request(f"{url}/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.loads(r.read())["choices"][0]["message"]["content"]


def from_stored(path: Path, per_prompt: int, safety_extra: int) -> list[dict]:
    d = json.loads(path.read_text(encoding="utf-8"))
    blocks = []
    order = {"safety_critical": 0, "honesty": 1, "diagnosis": 2, "instruction": 3,
             "agronomy": 4, "robustness": 5, "multiturn": 6, "systems": 7}
    for r in sorted(d["results"], key=lambda r: order.get(r["category"], 9)):
        n = per_prompt + (safety_extra if r["category"] == "safety_critical" else 0)
        blocks.append({
            "id": r["id"], "category": r["category"], "prompt": r["prompt"],
            "followup": r.get("followup"),
            "harness_score": r["score"], "passed": r["passed"], "n": r["n"],
            "answers": [s["text"] for s in r["samples"][:n]],
            "fails": sorted({f for s in r["samples"] for f in s["fails"]}),
        })
    return blocks


def from_live(url: str, prompts: list[str], per_prompt: int) -> list[dict]:
    blocks = []
    for i, p in enumerate(prompts, 1):
        print(f"  [{i}/{len(prompts)}] {p[:60]}...", file=sys.stderr)
        answers = []
        for _ in range(per_prompt):
            try:
                answers.append(ask(url, p))
            except (urllib.error.URLError, OSError) as exc:
                answers.append(f"<request failed: {exc}>")
        blocks.append({"id": f"live_{i:02d}", "category": "live", "prompt": p,
                       "followup": None, "harness_score": None,
                       "passed": None, "n": per_prompt,
                       "answers": answers, "fails": []})
    return blocks


def render(blocks: list[dict], title: str) -> str:
    out = [f"# {title}", ""]
    out.append("Raw model output for human review. The harness score is shown where")
    out.append("available but should be treated as a hint, not a verdict - it grades")
    out.append("string patterns and has been wrong about phrasing before.")
    out.append("")
    out.append("For each answer, judge: is it factually correct? is it safe? does it")
    out.append("obey any stated constraint? would a farmer act on it and be fine?")
    out.append("")

    cur = None
    for b in blocks:
        if b["category"] != cur:
            cur = b["category"]
            out += ["", "---", "", f"## {cur}", ""]

        score = ("n/a" if b["harness_score"] is None
                 else f"{b['harness_score']:.2f} ({b['passed']}/{b['n']} samples clean)")
        out.append(f"### {b['id']}")
        out.append("")
        out.append(f"**Harness score:** {score}")
        if b["fails"]:
            out.append("")
            out.append("**Rules tripped (may be wrong):**")
            for f in b["fails"][:4]:
                out.append(f"- `{f[:160]}`")
        out.append("")
        out.append(f"**PROMPT:** {b['prompt']}")
        if b["followup"]:
            out.append("")
            out.append(f"**FOLLOW-UP:** {b['followup']}")
        out.append("")
        for j, a in enumerate(b["answers"], 1):
            label = f"**ANSWER {j}:**" if len(b["answers"]) > 1 else "**ANSWER:**"
            out.append(label)
            out.append("")
            out.append("```")
            out.append(a.strip())
            out.append("```")
            out.append("")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("result", type=Path, nargs="?", help="an eval result JSON")
    ap.add_argument("--live", action="store_true", help="query a running server instead")
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--prompts", type=Path, help="one prompt per line, for --live")
    ap.add_argument("--per-prompt", type=int, default=1,
                    help="answers to include per prompt (default 1)")
    ap.add_argument("--safety-extra", type=int, default=2,
                    help="extra answers for safety_critical prompts, to expose inconsistency")
    ap.add_argument("-o", "--out", type=Path, default=Path("review.md"))
    args = ap.parse_args()

    if args.live:
        if not args.prompts or not args.prompts.exists():
            print("--live needs --prompts <file>", file=sys.stderr)
            return 1
        lines = [l.strip() for l in args.prompts.read_text(encoding="utf-8").splitlines()
                 if l.strip() and not l.startswith("#")]
        blocks = from_live(args.url, lines, args.per_prompt)
        title = f"AgriLLM review - live, {len(lines)} prompts"
    else:
        if not args.result or not args.result.exists():
            print("give an eval result JSON, or use --live", file=sys.stderr)
            return 1
        blocks = from_stored(args.result, args.per_prompt, args.safety_extra)
        title = f"AgriLLM review - {args.result.parent.parent.name}/{args.result.stem}"

    text = render(blocks, title)
    args.out.write_text(text, encoding="utf-8")
    words = len(text.split())
    print(f"wrote {args.out}  ({len(blocks)} prompts, ~{words} words, {len(text)} chars)")
    if words > 9000:
        print("  Large. Use --per-prompt 1 --safety-extra 1 to trim.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
