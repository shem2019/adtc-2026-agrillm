import json
import sys
from pathlib import Path
from collections import defaultdict

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from train.verify_corpus import shape_prefix  # noqa: E402

# These 3 files already pass verify_corpus.py's full-shape distinctness gate
# (every row has a unique shape()) but still fail the "worst 180-char prefix
# reused" gate. That gate is catching a real problem the shape() gate missed:
# a template-swap cluster where only a crop/breed noun differs, and that noun
# falls near the END of a long answer -- past the first 180 chars of the
# normalised shape. shape() treats these as distinct (because the tail
# differs); shape_prefix() correctly treats them as the same underlying fact
# padded out with cosmetic variation.
#
# Examples manually confirmed by reading full rows (not just shapes):
#   poultry.jsonl            -- "clean dry litter prevents coccidiosis/
#                                Salmonella" mechanism, only the chicken
#                                breed name swapped at the very end.
#   soil-fertility.jsonl      -- pH/micronutrient chlorosis mechanism
#                                template, only the crop name swapped.
#   water-conservation.jsonl -- Zai pit construction procedure template,
#                                only the crop name swapped at the end.
FILES = [
    "train/african/_clean/poultry.jsonl",
    "train/african/_clean/soil-fertility.jsonl",
    "train/african/_clean/water-conservation.jsonl",
]

MAX_PER_PREFIX = 1


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
            p = shape_prefix(ans)
            counts[p] += 1
            if counts[p] <= MAX_PER_PREFIX:
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
