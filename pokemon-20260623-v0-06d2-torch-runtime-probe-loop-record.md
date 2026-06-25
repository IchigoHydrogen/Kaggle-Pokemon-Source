# Pokemon Local Loop Record

## Version Goal

version: `pokemon-20260623-v0-06d2-torch-runtime-probe`

baseline: `pokemon-20260623-v0-06d1`

goal: Test whether a 300k-500k parameter PyTorch model is viable in production-form `main.py`.

hypothesis: A modest CPU PyTorch model can be loaded and run on every agent call without exhausting Kaggle-style per-action timeout and overage budgets. If this is true, future loops can evaluate real learned policy/value models directly instead of assuming distillation is required.

change_scope: Runtime feasibility only. Build a production-form archive with `main.py` importing torch and loading a model artifact, run per-call inference, and measure import/act/full-game latency. Do not claim gameplay improvement from random/untrained weights.

datasets:
- v0-06d1 final submission archive
- `pokemon-tcg-ai-battle` base data and `cg/` simulator

success_criteria:
- Production-form archive is generated with `main.py`, `deck.csv`, `cg/`, and a torch model artifact.
- `main.py` imports torch and loads a 300k-500k parameter model.
- Import succeeds repeatedly.
- Local games complete with zero illegal actions and zero exceptions.
- Per-call latency p99 and max are recorded.
- Estimated overage consumption is recorded for candidate `actTimeout` values.
- No automatic promotion to main agent strength is made from this feasibility run alone.

rollback_criteria:
- Torch import or model load fails.
- Local games produce errors or illegal actions.
- p99/max act latency is clearly unsafe.
- Full-game estimated overage approaches a 600-second match budget under plausible timeout assumptions.
- Archive packaging omits required files.

expected_artifacts:
- `pokemon-20260623-v0-06d2-torch-runtime-probe-submission-torch-probe.tar.gz`
- `pokemon-20260623-v0-06d2-torch-runtime-probe-runtime_report.json`
- `pokemon-20260623-v0-06d2-torch-runtime-probe-latency_rows.parquet`
- `pokemon-20260623-v0-06d2-torch-runtime-probe-promotion-decision.json`

## Pre-Implementation Self-Review

experiment_plan: Build a real submission-style torch runtime probe. The model will be loaded from a packaged `.pt` file at import and executed for every non-deck agent call over all legal options. The model output will not alter actions yet, so the result measures feasibility rather than strength.

implementation_plan: Create a standalone Python probe under `/kaggle/working`, extract v0-06d1 `main.py` and `deck.csv`, inject a small PyTorch MLP and option feature encoder into `main.py`, save a deterministic 300k-500k parameter state dict, package `main.py`, `deck.csv`, `cg/`, and the model file into a prefixed archive, import the generated agent repeatedly, and run local games while timing agent calls.

why_this_is_the_next_best_step: The prior assumption that distillation is necessary is unproven. A direct runtime feasibility test is cheaper and more decisive than debating architecture in the abstract.

what_would_make_this_result_untrustworthy: A dummy model that does not run every call would understate runtime cost. A benchmark that excludes import/model load would miss a real failure mode. Simple local opponents do not prove strength, so this run must not be promoted as a gameplay improvement.

expected_failure_modes:
- Torch import/model load latency is too high.
- Per-call inference is fine on average but has unsafe tail latency.
- Packaging a `.pt` model works locally but not under archive path assumptions.
- Extra archive size or import-time initialization becomes unacceptable.
- The model runs but the `cg` simulator catches illegal action regressions from source patching.

scope_guardrails:
- Do not change v0-06d1 tactical behavior.
- Do not train or tune for strength in this loop.
- Do not replace the promoted v0-06d1 agent based on this feasibility result alone.
- Record both favorable and unfavorable timing evidence.

validation_plan:
- Compile generated `main.py`.
- Build archive and inspect required files.
- Measure repeated import latency.
- Run local game durability with timed calls.
- Record p50/p95/p99/max per-call latency and full-game overage estimates for `actTimeout` candidates.
- Decide `promote`, `needs_followup`, or `reject` as a runtime feasibility result.

go_no_go: go. This tests a disputed premise directly and keeps gameplay strength claims out of scope.

## Promotion Decision

decision: promote

promotion_type: direct_torch_runtime_feasible

reason: A production-form archive with `main.py` importing torch and loading a `329,089` parameter model ran `96` local games with zero errors and zero illegal actions. Per-call latency was far below candidate `actTimeout` values and estimated overage consumption was zero for `0.25s`, `0.5s`, `1.0s`, `2.0s`, and `3.0s` thresholds.

hard_gates:
- model params: `329,089`
- archive size: `2,285,642` bytes
- archive includes `main.py`, `deck.csv`, `cg/api.py`, `cg/game.py`, and model `.pt`
- repeated import succeeded: max import time `0.02256s`
- games: `96`
- errors: `0`
- illegal actions: `0`
- timed target-agent calls: `5,850`
- options scored by torch: `43,393`
- act latency p50: `0.000253s`
- act latency p95: `0.000554s`
- act latency p99: `0.000771s`
- act latency max: `0.004081s`
- estimated overage at `actTimeout=0.25s`: `0.0s`
- estimated overage at `actTimeout=1.0s`: `0.0s`
- latency parquet written

baseline_comparison:
- v0-06d1 runtime torch: `False`
- v0-06d2 runtime torch: `True`
- v0-06d2 model file packaged in archive: `True`
- gameplay claim: none; random-weight probe intentionally measures runtime feasibility rather than strength

known_risks:
- This only proves runtime feasibility for one `329k` parameter CPU MLP shape, not every `500k` parameter architecture.
- The model is random and does not test policy quality.
- Kaggle production hardware and loaded libraries can differ from local Docker, so a guardrail using `remainingOverageTime` is still prudent.
- A real model may need more expensive feature extraction than this probe.

next_candidates:
- Train a real context-specific model if runtime is safe.
- Add fallback guardrails based on remainingOverageTime.
- Compare direct PyTorch inference versus distilled table on the same decisions.
