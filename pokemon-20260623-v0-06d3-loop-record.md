# Pokemon Local Loop Record

## Version Goal

version: `pokemon-20260623-v0-06d3`

baseline: `pokemon-20260623-v0-06d1`

goal: Produce a submit-ready PyTorch-in-runtime archive.

hypothesis: Since v0-06d2 showed a 329k-parameter PyTorch model can run safely in production form, we can produce a prefixed `pokemon-20260623-v0-06d3-submission.tar.gz` that imports torch, loads a packaged `.pt` model, runs inference each call, and still preserves v0-06d1 gameplay behavior in shadow mode.

change_scope: Packaging and runtime feasibility only. Keep v0-06d1 tactical decisions unchanged. The torch model runs in shadow mode and does not alter selected actions.

datasets:
- v0-06d1 final submission archive
- `pokemon-tcg-ai-battle` base data and `cg/` simulator

success_criteria:
- Prefixed submit-ready archive `pokemon-20260623-v0-06d3-submission.tar.gz` is generated in `/kaggle/working`.
- Archive contains `main.py`, `deck.csv`, `cg/api.py`, `cg/game.py`, and a packaged `.pt` model.
- Archive `main.py` imports torch and loads the packaged model.
- Local import and game durability pass with zero errors and zero illegal actions.
- Latency and overage estimates are recorded.
- Output artifacts live under `/kaggle/working/pokemon-20260623-v0-06d3/` with matching prefix.

rollback_criteria:
- Archive packaging fails or misses required files.
- Torch import/model load fails.
- Local games produce any exception or illegal action.
- Per-call latency is unsafe.
- The final tar.gz is not prefixed.

expected_artifacts:
- `pokemon-20260623-v0-06d3-submission.tar.gz`
- `pokemon-20260623-v0-06d3-runtime_report.json`
- `pokemon-20260623-v0-06d3-latency_rows.parquet`
- `pokemon-20260623-v0-06d3-promotion-decision.json`

## Pre-Implementation Self-Review

experiment_plan: Convert the v0-06d2 runtime probe into a submit-ready v0-06d3 archive. The archive should be safe to submit as a normal Kaggle `submission.tar.gz` equivalent, but with a prefixed filename for local accumulation.

implementation_plan: Copy the runtime probe script, change `RUN_PREFIX` to `pokemon-20260623-v0-06d3`, output the final archive as `pokemon-20260623-v0-06d3-submission.tar.gz`, preserve shadow-mode behavior, run durability checks, and record a promotion decision.

why_this_is_the_next_best_step: v0-06d2 proved feasibility but produced a probe-named archive. The next useful step is a clean submit-ready artifact with the final naming convention and validation record.

what_would_make_this_result_untrustworthy: Treating this as a strength improvement would be wrong because the model is still shadow-mode. It is a packaging/runtime milestone, not a gameplay milestone.

expected_failure_modes:
- Model path resolution fails inside archive import.
- Torch import succeeds locally but archive packaging misses the model file.
- Shadow inference adds latency tail risk.
- Output naming drifts from the required prefix.

scope_guardrails:
- Do not change v0-06d1 action behavior.
- Do not claim gameplay improvement.
- Do not tune deck or rule scores.
- Do not submit to Kaggle from this environment.

validation_plan:
- Compile the script and generated `main.py`.
- Build the archive.
- Inspect archive members.
- Run import and local durability checks.
- Record latency p50/p95/p99/max and overage estimates.
- Promote only as a submit-ready shadow-mode runtime artifact.

go_no_go: go. The loop converts a proven runtime probe into a clean submit-ready archive without mixing in a gameplay model.

## Promotion Decision

decision: promote

promotion_type: submit_ready_shadow_torch

reason: Generated a prefixed submit-ready archive that imports torch, loads a packaged `329,089` parameter model, runs shadow inference every agent call, and passes local durability with zero errors and zero illegal actions.

hard_gates:
- final archive: `/kaggle/working/pokemon-20260623-v0-06d3-submission.tar.gz`
- archive size: `2,285,542` bytes
- archive contains `main.py`, `deck.csv`, `cg/api.py`, `cg/game.py`, and `pokemon-20260623-v0-06d3-torch_probe_state.pt`
- archive missing required files: `[]`
- archive has `__pycache__`: `False`
- archive has `.pyc/.pyo`: `False`
- `main.py` imports torch and calls `torch.load`
- model params: `329,089`
- repeated import max: `0.017835s`
- games: `96`
- errors: `0`
- illegal actions: `0`
- timed target-agent calls: `5,789`
- options scored by torch: `43,046`
- act latency p50: `0.000236s`
- act latency p95: `0.000502s`
- act latency p99: `0.000676s`
- act latency max: `0.006011s`
- estimated overage at `actTimeout=0.25s`: `0.0s`
- latency parquet written

baseline_comparison:
- v0-06d1 behavior preserved by design: shadow mode
- v0-06d3 runtime torch: `True`
- gameplay claim: none; model output does not alter actions

known_risks:
- This artifact is submit-ready but shadow-mode. It does not yet use a trained NN to alter actions.
- Kaggle production runtime may differ from local Docker, so `remainingOverageTime` fallback remains advisable before enabling a real model.
- A real trained model may have more expensive feature extraction.

## External Submission Note

The user reported that `pokemon-20260623-v0-06d3-submission.tar.gz` was submitted on the official Kaggle side and the match completed in about `2` minutes. Treat this as practical evidence that a shadow-mode PyTorch runtime with roughly this parameter scale is acceptable in the official environment.

Implication for future loops: `~329k` parameters is a reasonable starting scale for direct PyTorch runtime models. Continue to keep latency/overage guardrails, but do not assume distillation is required purely for runtime reasons at this scale.

next_candidates:
- Train a real PyTorch reranker and enable it behind confidence and timing guards.
- Add `remainingOverageTime` fallback to skip neural inference when budget is low.
- Benchmark a near-500k architecture before enabling all-context neural policy.
