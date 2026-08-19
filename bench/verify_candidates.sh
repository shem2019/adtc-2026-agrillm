#!/usr/bin/env bash
# Pre-flight check: confirm every candidate URL exists before the sweep starts.
#
# HuggingFace GGUF repos vary a lot in naming, and some official repos publish
# only one quantization level (Qwen/Qwen3-1.7B-GGUF ships Q8_0 and nothing else).
# A 404 mid-sweep wastes a slot silently. This catches it in ~10 seconds.
#
#   ./verify_candidates.sh

set -uo pipefail
cd "$(dirname "$0")"

ok=0
bad=0

printf "%-22s %-10s %s\n" "CANDIDATE" "STATUS" "SIZE"
printf "%s\n" "----------------------------------------------------------------"

while IFS=$'\t' read -r label repo filename params; do
  case "$label" in ''|'#'*) continue ;; esac
  [ -n "${filename:-}" ] || continue

  url="https://huggingface.co/${repo}/resolve/main/${filename}"

  # -I is a HEAD request: asks for the headers only, downloads no data.
  # -L follows the redirect HuggingFace issues to its CDN.
  headers=$(curl -sIL --max-time 20 "$url" 2>/dev/null)
  code=$(printf "%s" "$headers" | awk '/^HTTP/{c=$2} END{print c}')
  bytes=$(printf "%s" "$headers" | awk -F': ' 'tolower($1)=="content-length"{v=$2} END{gsub(/\r/,"",v); print v}')

  if [ "${code:-000}" = "200" ]; then
    if [ -n "${bytes:-}" ] && [ "$bytes" -gt 0 ] 2>/dev/null; then
      size=$(awk -v b="$bytes" 'BEGIN{printf "%.2f GB", b/1073741824}')
    else
      size="?"
    fi
    printf "%-22s \033[32m%-10s\033[0m %s\n" "$label" "OK" "$size"
    ok=$((ok+1))
  else
    printf "%-22s \033[31m%-10s\033[0m %s\n" "$label" "HTTP ${code:-fail}" "$repo/$filename"
    bad=$((bad+1))
  fi
done < candidates.tsv

echo
echo "$ok reachable, $bad broken."
if [ "$bad" -gt 0 ]; then
  echo
  echo "For each broken one: open https://huggingface.co/<repo>/tree/main in a"
  echo "browser, find the real filename, and correct the third column in"
  echo "candidates.tsv. Or just delete the line — the sweep works fine with fewer"
  echo "candidates, and the four VERIFIED entries are enough to make a decision."
fi
