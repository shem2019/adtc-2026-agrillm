# Training pipeline — full fine-tune on a rented H100

Everything needed to take `train/african/_clean/*.jsonl` (6,703 verified rows)
to a scored, shippable Q4_K_M GGUF. Written so the GPU meter runs for the
shortest possible time: nothing here needs thinking about while it's billing.

## The short version

```bash
# on the rented box, from a clean checkout - no manual edits required
git clone https://github.com/shem2019/adtc-2026-agrillm.git
cd adtc-2026-agrillm

bash train/pipeline/setup_gpu.sh                 # ~15-20 min, once, idempotent
source .venv-train/bin/activate
bash train/pipeline/preflight.sh                 # ~30 s, must be all green

tmux new -s train                                # long runs, survive a dropped ssh
bash train/pipeline/run_pipeline.sh              # single stage, our corpus
#   or
bash train/pipeline/run_two_stage.sh             # external corpus, then ours
```

**Nothing tracked by git needs editing on the box.** `setup_gpu.sh` installs the
CUDA toolkit if absent, verifies the C++ toolchain, guards torch against being
downgraded to a CPU build by llama.cpp's requirements, and writes the pinned
base-model commit to `base_model_pin.json`, which the training scripts read.
`sft_config.yaml` stays at `base_model_revision: null` on purpose - hard-coding
the SHA there meant editing a tracked file on every box, which then conflicted
with every `git pull`. A manual step you have to remember is a reproducibility
bug, and Gate 2 Section 3.2 has organizers re-running this.

`preflight.sh` checks the things that have each cost real GPU time: a CPU-only
torch, a chat-template API whose return type changed, an `sdpa` attention
setting that produces non-finite gradients, a missing GGUF converter, an
unpinned base model, and the loss mask itself. Thirty seconds, and strictly
cheaper than finding any of them mid-run.

Then read the ranked table at the end, pick the winner, and:

```bash
CANDIDATE_GGUF=train/pipeline/runs/fullft-lr1e-5/gguf/checkpoint-406-Q4_K_M.gguf \
  bash train/pipeline/run_pipeline.sh --stage verify
```

## What each file does

| File | Role |
|---|---|
| `setup_gpu.sh` | Bootstraps the box: CUDA torch, transformers, llama.cpp built with CUDA, base model downloaded and its commit SHA pinned to `base_model_pin.json`. Idempotent. |
| `prompts.py` | **Single source of truth for system-prompt strings.** Imported by both the data prep and the template patcher so the string trained under and the string shipped can't drift. |
| `prepare_sft_data.py` | 25 JSONL files → tokenised tensors. Applies the Qwen ChatML template, masks loss to assistant tokens only, draws from the system-prompt mix, writes `manifest.json` with per-file SHA256 and `sample_rendered.txt` for human inspection. |
| `test_masking.py` | Proves the loss mask is right, including a negative control. Runs anywhere; `--real` uses the genuine tokenizer. |
| `sft_config.yaml` | Hyperparameters, with the reasoning for each recorded inline. Doubles as the Gate 2 Section 3.1 "training config" artefact. |
| `train_sft.py` | Full-parameter SFT via plain `transformers.Trainer`. Saves every epoch, writes `run_record.json` (loss history, step count, wall clock, GPU). |
| `set_chat_template_default.py` | Rewrites the template's injected default system prompt from Qwen's to AgriLLM's. Backs up, idempotent, `--check` and `--restore`. |
| `export_gguf.sh` | HF checkpoint → f16 GGUF → Q4_K_M. Repairs missing tokenizer files, patches the default prompt, verifies GGUF magic bytes, confirms both the ChatML template and the AgriLLM default survived into the binary, prints SHA256. |
| `select_checkpoint.sh` | Exports and scores **every** checkpoint with `eval/run_eval.py`, ranks them, refuses to crown a winner that has any safety-critical failure. |
| `compare_quants.sh` | Scores the winner at Q4_K_M / Q5_K_M / Q6_K / f16 so the size-vs-accuracy trade is made with numbers. |
| `run_pipeline.sh` | Orchestrates all seven stages; `--stage <name>` runs just one. |
| `capture_provenance.py` | Assembles `provenance/` for Gate 2 Section 3.1 and drafts the Model Provenance section for REPORT.md. |

## Identity: what the model says it is

Qwen2.5's chat template silently injects `You are Qwen, created by Alibaba
Cloud. You are a helpful assistant.` whenever the caller sends **no** system
message — which is exactly how the ADTC harness calls it. Left alone, the
shipped model is told it's a general-purpose Alibaba assistant on every scored
call.

`set_chat_template_default.py` replaces that injected default with:

> You are AgriLLM, an offline farming advisor for African smallholders, built on
> Qwen2.5-1.5B. Give practical, correct advice in plain language. Never state a
> pesticide or veterinary dose; point the person to the product label and their
> agrovet or veterinary officer. Say plainly when you do not know something
> rather than guessing.

It names the Qwen base deliberately. Gate 2 Section 3.1 requires disclosing the base
model, so a model that claims to be wholly homegrown contradicts your own
provenance folder. `identity-scope.jsonl` (30 rows) teaches the same honest
self-description, plus scope boundaries and the fact that it is **text-only** and
cannot see photographs.

Training doesn't use one fixed system prompt, because a model that is only safe
*when told to be* isn't safe. The mix is 45% the AgriLLM default (matching a
prompt-less grader call exactly), 20% Qwen's original default (insurance if the
patch doesn't propagate), 20% AgriLLM paraphrases, 15% a bland "You are a helpful
assistant." That last share is the control: it proves the behaviour isn't keyed
to specific magic words.

## Three decisions worth understanding before you run it

**1. Loss is computed on assistant tokens only.** The Round 1 checkpoint was
trained on raw `Q:/A:` completion text and then served through a ChatML
template it had never seen — that mismatch is why it drifted, rambled, and blew
a stated 150-word limit by 90 words. Everything here goes through
`tokenizer.apply_chat_template`, and `prepare_sft_data.py` verifies
prefix-consistency per row rather than trusting the mask.

**Before you launch a run, read `train/pipeline/data/sample_rendered.txt`.** It
prints the full rendered sequence and, separately, exactly which tokens loss is
computed on. If that second block contains anything other than assistant text,
stop — that is the single most expensive bug available here.

**2. The learning rate is 1e-5, not 5e-5.** Full fine-tuning updates every
weight and forgets base-model capability much faster than LoRA does. 5e-5 is a
reasonable LoRA rate and roughly 5× too hot for full FT. The default sweep tries 5e-6,
1e-5 and 2e-5. If eval comes back fluent-but-hollow — correct shape, lost general
knowledge — go *down* to 7e-6 before adding epochs.

**3. Checkpoints are chosen behaviourally, never by loss.** Validation loss
cannot see that "induce vomiting" is a disqualifying answer, that
*Heterorhabdium sheathi* is not a real species, or that a reply ran past a word
limit. `eval/run_eval.py` checks all three with string rules, samples each
prompt N times, and scores safety items on their **worst** sample — because the
most damning Round 1 finding was inconsistency: the model refused an Imidacloprid
dose for the human judge and then quoted "100 to 150 ppm" in the automated run.

## Memory and time

Full FT of 1.54B params on one H100 80GB:

```
fp32 master weights           ~5.8 GB
fp32 gradients                ~5.8 GB
AdamW states (2 x fp32)      ~11.5 GB
logits @ bs4, seq1024, fp32   ~2.3 GB   <-- vocab is 151,936; this dominates
activations                   ~2-6 GB
------------------------------------
working set                  ~28-32 GB of 80
```

The logits line is the one worth internalising. For a 1.5B model with a 152k
vocabulary, the vocab projection is a bigger single allocation than anything in
the model itself: `batch x seq x 151936`. At batch 16 that tensor alone is
~9.3 GB in fp32, which is how the first attempt at this run managed to OOM on an
80 GB card. batch_size is 4 with grad_accum 8 for that reason, not for
throughput.

Corpus p99 is ~562 tokens, so `max_len: 1024` clears every row. ~6,500 training
examples at effective batch 32 ≈ 204 steps/epoch, ~816 steps for 4 epochs — a
few minutes of actual compute. Training is not the expensive part; the sweep,
exports and evals dominate wall clock.

**Where the budget goes, and why.** The default settings spend compute on
reducing *selection* risk rather than on training longer, because the binding
constraint on quality here is the corpus (6,703 rows), not GPU hours. Training
past the point of fit just overfits. So:

- three learning rates (5e-6 / 1e-5 / 2e-5) rather than one guess
- four epochs with per-epoch checkpoints = 12 candidates
- 8 eval samples per prompt instead of 3 — at temperature 0.8 a 3-sample score
  is noisy enough to rank the wrong checkpoint first, and a bad pick wastes
  everything upstream
- a quantisation sweep on the winner, because Round 1 shipped Q4_K_M purely for
  size and nobody ever measured what the 4-bit squeeze cost in accuracy

That lands around **3-4 hours, roughly $12-16** at $3.822/hr. Trim `--lrs` or
`--samples` if you want it cheaper; raise `--samples` if two checkpoints finish
within a couple of points of each other.

## If something goes wrong

| Symptom | Likely cause |
|---|---|
| `sample_rendered.txt` shows question text in the loss block | Chat template changed upstream; the prefix-consistency check in `encode()` should have caught it — investigate before training. |
| OOM | Card isn't actually 80GB, or `batch_size` was raised. Halve `batch_size`, double `grad_accum`. |
| GGUF has no ChatML template | `export_gguf.sh` prints `chatml tmpl: NO`. The checkpoint dir lost its tokenizer; delete the GGUF and re-run the export so it re-copies from the pinned base. |
| Every checkpoint has safety-critical failures | Don't ship any. Lower the LR, or check the loss mask. A model that fails these after training on 166 poisoning + 167 dose-refusal rows has a pipeline bug, not a data problem. |
| `llama-server` won't start during selection | See `runs/*/eval/<ckpt>-server.log`. Usually a bad GGUF from an interrupted export — delete it and re-run. |

## Reproducing without a GPU

`prepare_sft_data.py` runs anywhere (CPU-only, needs `transformers`). It's worth
running locally first to eyeball `sample_rendered.txt` before you start paying
for a GPU — that check costs nothing and catches the expensive bug.
