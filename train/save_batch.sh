#!/usr/bin/env bash
# Append a Gemini batch from the clipboard to a named topic file.
#
#   ./train/save_batch.sh maize-pests        # appends to train/african/maize-pests.jsonl
#   ./train/save_batch.sh maize-pests --new  # start that topic over
#   ./train/save_batch.sh --list             # show progress against every target
#   ./train/save_batch.sh maize-pests --quick  # skip validation (faster mid-run)
#
# APPEND, not new files per batch: one file per topic means the validator's
# per-file counts ARE your per-topic counts, so you can see which topics still
# need work instead of counting scattered -a/-b/-c fragments.
#
# pbpaste reads the macOS clipboard directly — nothing is pasted into the shell,
# which matters because JSONL is full of quotes and backslashes.

set -euo pipefail
cd "$(dirname "$0")/.."

DIR="train/african"
mkdir -p "$DIR"

# Topic slug -> target count. Mirrors the table in GEMINI_PROMPT.md.
TOPICS="maize-pests:1200
maize-agronomy:800
beans-legumes:800
cassava-sweetpotato:700
banana:500
coffee-tea:500
potato-horticulture:700
sorghum-millet-groundnut:500
soil-fertility:900
water-conservation:700
climate-seasons:600
post-harvest:700
livestock:900
poultry:500
economics-extension:500
swahili:2500"

target_for() { echo "$TOPICS" | grep "^$1:" | cut -d: -f2; }
count_of()  { [ -f "$DIR/$1.jsonl" ] && grep -c . "$DIR/$1.jsonl" || echo 0; }

show_progress() {
  local grand=0 gtarget=0
  printf "\n  %-28s %8s %8s %7s\n" "TOPIC" "HAVE" "TARGET" "DONE"
  printf "  %s\n" "-----------------------------------------------------"
  while IFS=: read -r slug target; do
    [ -n "$slug" ] || continue
    local have; have=$(count_of "$slug")
    grand=$((grand + have)); gtarget=$((gtarget + target))
    local pct=$((have * 100 / target))
    local bar=""; [ "$pct" -ge 100 ] && bar=" done"
    printf "  %-28s %8s %8s %6s%%%s\n" "$slug" "$have" "$target" "$pct" "$bar"
  done <<< "$TOPICS"
  printf "  %s\n" "-----------------------------------------------------"
  printf "  %-28s %8s %8s %6s%%\n" "TOTAL" "$grand" "$gtarget" \
         "$((grand * 100 / gtarget))"
}

if [ "${1:-}" = "--list" ]; then show_progress; exit 0; fi

TOPIC="${1:-}"
if [ -z "$TOPIC" ]; then
  echo "usage: ./train/save_batch.sh <topic-slug> [--new|--quick]"
  echo "       ./train/save_batch.sh --list"
  echo
  echo "topics:"
  echo "$TOPICS" | sed 's/:/  -> target /' | sed 's/^/  /'
  exit 2
fi

if ! echo "$TOPICS" | grep -q "^${TOPIC}:"; then
  echo "✗ unknown topic '$TOPIC'. Run --list to see valid slugs."
  exit 2
fi

DEST="$DIR/${TOPIC}.jsonl"
MODE="${2:-}"
[ "$MODE" = "--new" ] && : > "$DEST"

BEFORE=$(count_of "$TOPIC")
TMP=$(mktemp)
pbpaste > "$TMP"

if [ ! -s "$TMP" ]; then
  echo "✗ clipboard empty — nothing written."; rm -f "$TMP"; exit 1
fi

FIRST=$(head -c 1 "$TMP")
if [ "$FIRST" != "{" ] && [ "$FIRST" != '`' ]; then
  echo "  ⚠ first character is '$FIRST', expected '{' — Gemini added a preamble."
  echo "    Saving anyway; the validator will reject the junk lines."
fi

# Ensure a trailing newline so appended batches never fuse two JSON objects
# onto one line — that would corrupt the last entry of one batch and the first
# of the next, every single time.
[ -s "$DEST" ] && [ "$(tail -c1 "$DEST" | wc -l)" -eq 0 ] && echo >> "$DEST"
cat "$TMP" >> "$DEST"
[ "$(tail -c1 "$DEST" | wc -l)" -eq 0 ] && echo >> "$DEST"
rm -f "$TMP"

AFTER=$(count_of "$TOPIC")
TARGET=$(target_for "$TOPIC")
echo "✓ ${TOPIC}: +$((AFTER - BEFORE)) lines  (now $AFTER of $TARGET target)"

REMAIN=$((TARGET - AFTER))
if [ "$REMAIN" -gt 0 ]; then
  echo "  $REMAIN to go — next: \"continue, 250 more, do not repeat anything above\""
else
  echo "  target reached. Move to the next topic."
fi

[ "$MODE" = "--quick" ] && exit 0
echo
python3 train/validate_african.py
show_progress
