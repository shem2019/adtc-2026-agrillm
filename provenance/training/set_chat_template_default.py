#!/usr/bin/env python3
"""Replace the default system prompt baked into a checkpoint's chat template.

    python3 train/pipeline/set_chat_template_default.py <checkpoint_dir>
    python3 train/pipeline/set_chat_template_default.py <checkpoint_dir> --check

Qwen2.5's stock template contains, in two places:

    'You are Qwen, created by Alibaba Cloud. You are a helpful assistant.'

injected whenever the caller supplies no system message. The ADTC grading
harness calls the model with no system message, so on every scored call the
shipped model is being told it is a general-purpose Alibaba assistant. That is
the opposite of what we trained it to be.

convert_hf_to_gguf.py copies `chat_template` out of tokenizer_config.json into
GGUF metadata verbatim, and llama.cpp renders from that. So editing this file in
the checkpoint directory *before* conversion is what makes the change reach the
shipped artefact. Run it after training and before export_gguf.sh -
export_gguf.sh already calls it for you.

The replacement string comes from prompts.py, which is also what
prepare_sft_data.py trains under, so the model meets the same text at inference
that it saw in training.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from prompts import AGRI_DEFAULT  # noqa: E402

QWEN_DEFAULT = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("--check", action="store_true",
                    help="report status and exit without writing")
    ap.add_argument("--restore", action="store_true",
                    help="put the original template back from the .bak")
    args = ap.parse_args()

    # transformers >= ~4.49 saves the chat template to a standalone
    # chat_template.jinja rather than embedding it in tokenizer_config.json.
    # Both layouts exist in the wild and llama.cpp's converter reads either, so
    # find whichever is present - and patch BOTH if both are, since we cannot be
    # sure which one the converter will pick up.
    cfg_path = args.checkpoint / "tokenizer_config.json"
    jinja_path = args.checkpoint / "chat_template.jinja"

    targets: list[tuple[str, Path]] = []
    if jinja_path.exists():
        targets.append(("jinja", jinja_path))
    if cfg_path.exists():
        try:
            if isinstance(json.loads(cfg_path.read_text(encoding="utf-8")).get("chat_template"), str):
                targets.append(("json", cfg_path))
        except json.JSONDecodeError:
            pass

    if not targets:
        print(f"no chat template found in {args.checkpoint}", file=sys.stderr)
        print("  looked for: chat_template.jinja, and a chat_template key in "
              "tokenizer_config.json", file=sys.stderr)
        print("  Run export_gguf.sh, which copies the tokenizer from the pinned base first.",
              file=sys.stderr)
        return 1

    def read_template(kind: str, path: Path) -> str:
        if kind == "jinja":
            return path.read_text(encoding="utf-8")
        return json.loads(path.read_text(encoding="utf-8"))["chat_template"]

    def write_template(kind: str, path: Path, text: str) -> None:
        if kind == "jinja":
            path.write_text(text, encoding="utf-8")
        else:
            cfg = json.loads(path.read_text(encoding="utf-8"))
            cfg["chat_template"] = text
            path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n",
                            encoding="utf-8")

    if args.restore:
        restored = 0
        for kind, path in targets:
            bak = path.with_suffix(path.suffix + ".bak")
            if bak.exists():
                shutil.copy2(bak, path)
                print(f"restored {path.name} from {bak.name}")
                restored += 1
        if not restored:
            print("no backups found", file=sys.stderr)
            return 1
        return 0

    template = read_template(*targets[0])
    n_qwen = template.count(QWEN_DEFAULT)
    n_agri = template.count(AGRI_DEFAULT)
    print(f"template source: {', '.join(p.name for _, p in targets)}")

    if args.check:
        print(f"checkpoint : {args.checkpoint}")
        print(f"  Qwen default occurrences    : {n_qwen}")
        print(f"  AgriLLM default occurrences : {n_agri}")
        print("  status:", "PATCHED" if n_agri and not n_qwen else
              ("UNPATCHED" if n_qwen else "UNKNOWN - template does not match either"))
        return 0

    if n_agri and not n_qwen:
        print(f"already patched ({n_agri} occurrence(s)); nothing to do")
        return 0

    if not n_qwen:
        print("! Could not find Qwen's default string in the template.", file=sys.stderr)
        print("  Upstream may have changed the template. Inspect it with:", file=sys.stderr)
        print(f"    cat {targets[0][1]}", file=sys.stderr)
        return 1

    patched_text = template.replace(QWEN_DEFAULT, AGRI_DEFAULT)
    for kind, path in targets:
        bak = path.with_suffix(path.suffix + ".bak")
        if not bak.exists():
            shutil.copy2(path, bak)
        write_template(kind, path, patched_text)
        print(f"patched {n_qwen} occurrence(s) in {path.name}")
    print(f"  new default: {AGRI_DEFAULT[:78]}...")

    # Prove it renders. Worth doing here rather than discovering it in the GGUF.
    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(args.checkpoint)
        rendered = tok.apply_chat_template(
            [{"role": "user", "content": "Who are you?"}],
            tokenize=False,
            add_generation_prompt=True,
        )
        first_line = rendered.split("<|im_end|>")[0].replace("<|im_start|>system\n", "")
        print("\n  render check, no system message supplied:")
        print(f"    {first_line.strip()[:100]}...")
        if AGRI_DEFAULT[:40] not in rendered:
            print("  ! AgriLLM default did NOT appear in the render", file=sys.stderr)
            return 1
        print("  ok: a prompt-less call now receives the AgriLLM system prompt")
    except ImportError:
        print("\n  (transformers not installed here - render check skipped)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
