# Gate 2 submission checklist

Checked against the repo on 21 September 2026, deadline 22 September. Items are
from the Gate 2 guidelines Section 4. Anything marked OPEN needs action.

Re-verified on 22 September against the template checklist at template commit
`42521b4` (21 September) and profiler commit `7f117dd` (still the latest):
repo public; `download_model.sh` differs from the template only in `MODEL_FILE`
and `MODEL_URL`; the pinned URL serves 986,048,608 bytes with SHA256
`ad7e079f…`, matching `provenance/checksums.txt`; no weights tracked; exactly 2
test prompts; no `git_commit_sha` key in `metadata.json`.

## Done

| Item | Evidence |
|---|---|
| Template-compliant repo | `metadata.json`, `download_model.sh`, `REPORT.md` present; `model/` and `*.gguf` in `.gitignore`; no weights tracked in git |
| `metadata.json` complete | All required fields present, exactly 2 test prompts, no placeholder values remaining |
| Model Provenance section | REPORT.md Section 3 — base model, commit SHA, method, both datasets, before/after |
| Git commit SHA in metadata | `provenance.base_model_commit_sha` = `989aa7980e4cf806f80c7fef2b1adb7bc71aa306` |
| Fine-tuning method stated | `full_fine_tune`, two stages. Corrected from Round 1, where REPORT.md said LoRA while metadata said full fine-tune |
| Proof-of-training files | `provenance/` — 31 evaluation results, 7 per-step loss CSVs, trainer records, training scripts, dataset manifests with checksums, weight delta showing 99.9986% of parameters changed |
| Before/after comparison | `provenance/before_after.md` — 3 prompts plus full category table |
| SHA256 checksums | `provenance/checksums.txt` — base model, every published model file, full-precision checkpoints, source dataset, all 25 corpus files |
| `download_model.sh` static URL | Official template verbatim, only `MODEL_FILE` and `MODEL_URL` edited. URL is a plain literal pinned to HF commit `d84a627c…`, not a branch |
| `model_path` matches script | `model/adtc-agri-Q4_K_M.gguf` in both |
| Third-party attribution | `AI71ai/agrillm-train-146k` credited in REPORT.md Sections 3 and 7, REPORT_DETAILS.md Section 5, `metadata.json`, `provenance/README.md`, with licence, source SHA256 and filter counts |
| African use-case claim | `african_alpha_claim: true`, basis documented in REPORT.md Section 1. Round 1 had this `false` because it was conflated with the failed Swahili language claim |
| Profiler run on the shipped model | `provenance/benchmark/` — full participant-mode run on a clean 4 vCPU x86-64 instance, 21 September; metadata.json validates against the profiler schema at commit `7f117dd` |
| Model genuinely useful | REPORT.md Section 2 records that the 0.5B model scores better on Efficiency and Performance and was rejected on answer quality |
| Updated video | https://youtu.be/oGjm1zTiEGA — exactly 2:00, uploaded 22 September; linked from README.md |
| REPORT.md follows the template | Sections 1–5 are the template's Problem, Design Decisions, Model Provenance, Constraints, Benchmarks, in that order; about 2,000 words. Full detail moved to `REPORT_DETAILS.md` |

## Open

| Item | What is needed |
|---|---|
| **Name consistency** | `metadata.json` says "Shem Kinyanjui Njuguna"; the call booking says "Shem Njuguna". Confirm both match the ADTF portal registration. |
| **Eligibility** | Age, funding under $25,000, project under 12 months, resident in an eligible country — self-confirm. |

## Known and disclosed

Not blockers, but a judge will find them, so they are stated in the report rather
than left to be discovered.

- Swahili fails; the language bonus is not claimed (REPORT.md Section 6).
- The shipped model advises keeping contaminated clothing on after a pesticide
  spill. It should come off. Needs a corpus fix.
- The internal test over-scores by 13–18 points against a human reading the same
  answers, and has a ±6 point noise margin.
- The 57.9% figure for the first two-stage attempt has no preserved result file;
  the retry reused the run name and overwrote it.
- The quantisation sweep was run for the single-stage models, not re-run for the
  shipped two-stage model.
