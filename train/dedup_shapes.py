import json
import sys
from pathlib import Path
from collections import defaultdict

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from train.verify_corpus import shape  # noqa: E402

FILES = [
    "train/african/_clean/economics-extension.jsonl",
    "train/african/_clean/potato-horticulture.jsonl",
    "train/african/_clean/sorghum-millet-groundnut.jsonl",
    "train/african/_clean/poultry.jsonl",
    "train/african/_clean/water-conservation.jsonl",
    "train/african/_clean/soil-fertility.jsonl",
]

MAX_PER_SHAPE = 1


def get_answer(row: dict) -> str:
    if "messages" in row:
        return row["messages"][-1]["content"]
    return row.get("answer", "")


def main():
    results = []
    for rel in FILES:
        path = REPO_ROOT / rel
        raw_lines = [
            line for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        original_count = len(raw_lines)

        counts = defaultdict(int)
        kept_lines = []
        parse_failures = 0
        for line in raw_lines:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                # Keep malformed lines as-is (verify_corpus would flag them
                # separately); don't drop data we can't classify.
                kept_lines.append(line)
                parse_failures += 1
                continue
            ans = get_answer(row)
            s = shape(ans)
            counts[s] += 1
            if counts[s] <= MAX_PER_SHAPE:
                kept_lines.append(line)

        new_count = len(kept_lines)
        path.write_text(
            "\n".join(kept_lines) + "\n", encoding="utf-8"
        )
        results.append((rel, original_count, new_count, parse_failures))

    print(f"{'file':45s} {'before':>8s} {'after':>8s}")
    for rel, before, after, failures in results:
        note = f"  (parse_failures={failures})" if failures else ""
        print(f"{rel:45s} {before:8d} {after:8d}{note}")


if __name__ == "__main__":
    main()
