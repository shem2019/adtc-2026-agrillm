#!/usr/bin/env bash
# Export this project's Cowork session transcript into .context/ so Claude Code
# (or any later session) can read the full record rather than a summary.
#
#   bash tools/export_session.sh
#
# Writes:
#   .context/session-<id>.jsonl   the raw transcript, untouched
#   .context/session-<id>.md      the same thing rendered readable
#
# .context/ is gitignored — the transcript is working material, and this repo is
# public.

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$REPO_ROOT/.context"
mkdir -p "$OUT"

echo "Searching for Cowork transcripts…"
# Transcripts live under a per-run temp dir; the name is unpredictable, so find
# them by shape rather than by a hard-coded path.
mapfile -t FOUND < <(
  find /var/folders /private/tmp "$HOME/.claude" \
       -type f -name '*.jsonl' -path '*project*' 2>/dev/null \
    | xargs -r ls -t 2>/dev/null
)

if [ "${#FOUND[@]}" -eq 0 ]; then
  echo "No transcripts found."
  echo "In the Cowork app the file is under a temp directory that macOS may have"
  echo "cleared. If so, the conversation itself is still in the app."
  exit 1
fi

echo "Found ${#FOUND[@]} transcript file(s). Newest first:"
for f in "${FOUND[@]:0:8}"; do
  printf '  %8s  %s\n' "$(du -h "$f" | cut -f1)" "$f"
done

SRC="${1:-${FOUND[0]}}"
ID="$(basename "$SRC" .jsonl)"
cp "$SRC" "$OUT/session-$ID.jsonl"
echo
echo "Copied  -> .context/session-$ID.jsonl"

python3 - "$OUT/session-$ID.jsonl" "$OUT/session-$ID.md" <<'PY'
import json, sys, pathlib

src, dst = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])

def text_of(content):
    """Pull readable text out of whatever shape the content field takes."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for b in content:
            if not isinstance(b, dict):
                parts.append(str(b)); continue
            t = b.get("type")
            if t == "text":
                parts.append(b.get("text", ""))
            elif t == "thinking":
                continue                      # internal reasoning, skipped
            elif t == "tool_use":
                name = b.get("name", "tool")
                inp = json.dumps(b.get("input", {}), indent=2)[:1800]
                parts.append(f"\n**[tool: {name}]**\n```json\n{inp}\n```")
            elif t == "tool_result":
                c = b.get("content")
                s = text_of(c) if c is not None else ""
                parts.append(f"\n**[result]**\n```\n{s[:2500]}\n```")
            elif t == "image":
                parts.append("\n*[image]*")
        return "\n".join(p for p in parts if p)
    if isinstance(content, dict):
        return text_of(content.get("content", ""))
    return ""

rows, skipped = [], 0
for line in src.read_text(encoding="utf-8", errors="replace").splitlines():
    line = line.strip()
    if not line:
        continue
    try:
        d = json.loads(line)
    except json.JSONDecodeError:
        skipped += 1
        continue
    msg = d.get("message") or d
    role = msg.get("role") or d.get("type") or "?"
    body = text_of(msg.get("content", ""))
    if body.strip():
        rows.append((role, d.get("timestamp", ""), body))

out = [f"# Session transcript — {src.stem}",
       "",
       f"{len(rows)} messages."
       + (f" {skipped} unparseable lines skipped." if skipped else ""),
       "",
       "Internal reasoning blocks are omitted; everything said and every tool "
       "call is here.",
       "", "---", ""]
for role, ts, body in rows:
    head = role.upper() + (f"  ·  {ts}" if ts else "")
    out += [f"## {head}", "", body, "", "---", ""]

dst.write_text("\n".join(out), encoding="utf-8")
print(f"Rendered -> {dst.name}  ({len(rows)} messages, "
      f"{dst.stat().st_size//1024} KB)")
PY

echo
echo "Point Claude Code at it:"
echo "  \"read .context/session-$ID.md before we start\""
