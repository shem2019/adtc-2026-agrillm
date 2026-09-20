#!/usr/bin/env bash
# Verify a freshly set-up box can actually run the pipeline. Exits non-zero if not.
#
#   bash train/pipeline/preflight.sh
#
# Every check here exists because its absence cost real GPU time on a rented
# box. Running it takes about thirty seconds and is strictly cheaper than
# discovering any of these mid-run.

set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

PASS=0; FAIL=0
ok()   { printf '  \033[1;32m[ok]\033[0m   %s\n' "$*"; PASS=$((PASS+1)); }
bad()  { printf '  \033[1;31m[FAIL]\033[0m %s\n' "$*"; FAIL=$((FAIL+1)); }
note() { printf '         %s\n' "$*"; }

echo "preflight: $(git rev-parse --short HEAD 2>/dev/null || echo 'not a git repo')"
echo

# --- environment ------------------------------------------------------------
python3 - <<'PY' && ok "torch sees a CUDA device" || bad "torch cannot reach a GPU"
import sys, torch
if not torch.cuda.is_available(): sys.exit(1)
p = torch.cuda.get_device_properties(0)
print(f"         {p.name}, {p.total_memory/2**30:.1f} GB, torch {torch.__version__}, cuda {torch.version.cuda}")
if p.total_memory/2**30 < 38:
    print("         ! under 38GB - set batch_size 2 / grad_accum 16 in sft_config.yaml")
PY

# A CPU-only torch build is the specific failure that wasted an hour: it looks
# like a driver problem and is actually a dependency install having replaced the
# CUDA wheel.
python3 -c "
import sys, torch
sys.exit(0 if torch.version.cuda else 1)" \
  && ok "torch is a CUDA build" \
  || { bad "torch is CPU-only"; note "fix: pip uninstall -y torch && pip install torch"; }

python3 -c "import transformers, accelerate, yaml, numpy" 2>/dev/null \
  && ok "training stack imports" || bad "missing transformers/accelerate/yaml/numpy"

# --- the API-shape checks that have each broken a run -----------------------
python3 - <<'PY' && ok "apply_chat_template returns usable token ids" || bad "chat-template tokenisation broken"
import sys
sys.path.insert(0, "train/pipeline")
from transformers import AutoTokenizer
from prepare_sft_data import template_ids
import json, pathlib
pin = pathlib.Path("train/pipeline/base_model_pin.json")
cfg = json.loads(pin.read_text()) if pin.exists() else {}
tok = AutoTokenizer.from_pretrained(cfg.get("base_model", "Qwen/Qwen2.5-1.5B-Instruct"),
                                    revision=cfg.get("revision"))
ids = template_ids(tok, [{"role":"user","content":"hello"},
                         {"role":"assistant","content":"hi there"}], False)
if not ids or not isinstance(ids[0], int) or len(ids) < 5:
    sys.exit(1)
print(f"         {len(ids)} tokens, first={ids[0]}")
PY

python3 train/pipeline/test_masking.py >/dev/null 2>&1 \
  && ok "loss mask correct (test_masking.py)" || bad "loss mask test FAILED - do not train"

# --- llama.cpp --------------------------------------------------------------
BIN="${REPO_ROOT}/.llama.cpp/build/bin"
for b in llama-quantize llama-server llama-cli; do
  [ -x "${BIN}/${b}" ] && ok "${b} built" || bad "${b} missing"
done
python3 "${REPO_ROOT}/.llama.cpp/convert_hf_to_gguf.py" --help >/dev/null 2>&1 \
  && ok "convert_hf_to_gguf.py imports" \
  || { bad "GGUF converter cannot import"; note "fix: pip install gguf sentencepiece protobuf"; }

# --- config + data ----------------------------------------------------------
python3 - <<'PY' && ok "sft_config.yaml sane" || bad "sft_config.yaml problem"
import sys, yaml
c = yaml.safe_load(open("train/pipeline/sft_config.yaml"))
issues = []
if c.get("attn_implementation") != "eager":
    issues.append(f"attn_implementation is {c.get('attn_implementation')!r}, must be 'eager' "
                  "(sdpa + bf16 produces non-finite gradients on padded batches)")
if c["batch_size"] * c["grad_accum"] != 32:
    issues.append(f"effective batch is {c['batch_size']*c['grad_accum']}, expected 32")
for i in issues: print(f"         ! {i}")
print(f"         lr={c['learning_rate']} epochs={c['epochs']} "
      f"bs={c['batch_size']}x{c['grad_accum']} attn={c['attn_implementation']}")
sys.exit(1 if issues else 0)
PY

[ -f train/pipeline/base_model_pin.json ] \
  && ok "base model pinned: $(jq -r .revision train/pipeline/base_model_pin.json | cut -c1-16)..." \
  || bad "no base_model_pin.json - run setup_gpu.sh"

python3 train/verify_corpus.py train/african/_clean/*.jsonl 2>/dev/null | tail -1 | grep -q "files pass" \
  && ok "corpus gates: $(python3 train/verify_corpus.py train/african/_clean/*.jsonl 2>/dev/null | tail -1)" \
  || bad "corpus gate check failed"

echo
if [ "$FAIL" -eq 0 ]; then
  printf '\033[1;32mAll %d checks passed. Safe to train.\033[0m\n' "$PASS"
else
  printf '\033[1;31m%d check(s) FAILED, %d passed. Fix before spending GPU time.\033[0m\n' "$FAIL" "$PASS"
fi
exit $((FAIL > 0))
