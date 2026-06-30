# Pokemon Local Loop Record

## Version Goal

version: `pokemon-20260623-v0-05d5`

baseline: `pokemon-20260623-v0-05d4`

goal: Add split-aware UNKNOWN_0 policy evaluation for holdout/stress promotion decisions.

hypothesis: Joining UNKNOWN_0 prediction rows with deterministic episode splits and the distilled policy table will expose whether policy-table overrides are supported on holdout and stress slices, making future UNKNOWN_0 threshold changes less likely to overfit validation-only metrics.

change_scope: Evaluation/reporting only. Do not change deck choice, rule-agent behavior, UNKNOWN_0 policy thresholds, model architecture, model features, model training, or local evaluation game counts.

datasets:
- `top200-20260622-ranking`
- `pokemon-tcg-ai-battle-episodes-2026-06-22`
- `pokemon-tcg-ai-battle` base data

success_criteria:
- Notebook executes fully.
- `RUN_PREFIX` is `pokemon-20260623-v0-05d5`.
- Existing hard gates remain clean.
- Split-aware UNKNOWN_0 policy evaluation artifacts are generated and nonempty.
- Evaluation reports rows for `train`, `valid`, and `holdout`.
- Holdout policy-hit rows and UNKNOWN_0 decision counts are recorded in run summary.
- No gameplay behavior changes are made.

rollback_criteria:
- Notebook execution fails.
- Evaluation artifacts are missing or empty in full mode.
- Holdout split has zero UNKNOWN_0 policy-evaluable decisions.
- Any hard gate fails.
- Any required artifact path in run summary points to a missing file.

expected_artifacts:
- `pokemon-20260623-v0-05d5-unknown0_policy_split_eval.parquet`
- `pokemon-20260623-v0-05d5-unknown0_policy_stress_eval.parquet`
- `pokemon-20260623-v0-05d5-v05_run_summary.json`
- `pokemon-20260623-v0-05d5-promotion-decision.json`

## Pre-Implementation Self-Review

experiment_plan: Add diagnostics that evaluate the current UNKNOWN_0 policy table on the offline prediction rows by deterministic split and stress slices. This version should not change the policy table or submission behavior.

implementation_plan: Copy v0-05d4 to `pokemon-20260623-v0-05d5.ipynb`, update `EXPERIMENT_NAME` and `RUN_PREFIX`, add helper functions after UNKNOWN_0 policy-table construction to join `UNKNOWN0_MLP_VALID_PRED_DF` with `EPISODE_SPLIT_DF`, compute per-decision policy hits based on the distilled table keys, aggregate by split and stress slice, save the reports, and include row counts plus holdout coverage in `V05_RUN_SUMMARY`.

why_this_is_the_next_best_step: v0-05d4 pruned low-accuracy table entries, but promotion still relied on table-level validation metrics and noisy local smoke eval. Split-aware offline evaluation is needed before further UNKNOWN_0 threshold tuning.

what_would_make_this_result_untrustworthy: Using this diagnostic as a direct win-rate estimate would be untrustworthy. It only measures replay action agreement for policy-table candidate overrides, not actual game outcomes under altered play.

expected_failure_modes:
- Runtime policy key matching and offline key matching may differ.
- The prediction artifact is internally valid-split based, so split-aware reports are diagnostic rather than a perfect holdout training separation.
- Policy-hit coverage may be very sparse.
- Stress slices may be too coarse to isolate a specific matchup failure.

scope_guardrails:
- No deck list edits.
- No rule-agent edits.
- No UNKNOWN_0 threshold edits.
- No model retraining or feature edits.
- No local evaluation game-count edits.
- No Kaggle submission.

validation_plan:
- Run JSON/code-cell validation.
- Execute smoke mode.
- Execute full notebook.
- Confirm split-aware evaluation artifacts are nonempty.
- Confirm `holdout` appears in split evaluation.
- Confirm summary artifact paths exist.
- Confirm hard gates and submission archive checks still pass.

go_no_go: go. The plan is diagnostic-only, directly addresses the prior loop's known risk, and avoids mixing in another gameplay change.

## Promotion Decision

decision: promote

promotion_type: diagnostic_baseline

reason: Full notebook execution passed with clean hard gates and generated split-aware UNKNOWN_0 policy evaluation artifacts for train, valid, and holdout. Promote this as a diagnostics baseline for future UNKNOWN_0 threshold work, not as proof of gameplay improvement.

hard_gates:
- `RUN_MODE=full`
- `error_count=0`
- `unknown0_policy_decision_eval_rows=16446`
- `unknown0_policy_split_eval_rows=3`
- `unknown0_policy_stress_eval_rows=12`
- holdout UNKNOWN_0 policy-evaluable decisions: `2502`
- holdout policy-hit decisions: `1945`
- holdout policy-hit rate: `0.777378`
- holdout policy correctness on hits: `0.601542`
- submission archives are prefixed and contain `main.py`, `deck.csv`, `cg/api.py`, and `cg/game.py`
- archive inspection found no `__pycache__` and no torch-named files

baseline_comparison:
- No intended gameplay behavior change from `pokemon-20260623-v0-05d4`.
- Added `unknown0_policy_decision_eval`, `unknown0_policy_split_eval`, and `unknown0_policy_stress_eval`.
- Full run policy table entries: `59`.
- Local confirm summary: `74/80`, win rate `0.925`, errors `0`, illegal actions `0`.
- Selected variant runtime policy stats remain sparse in local smoke: `6` selections over `344` eligible UNKNOWN_0 calls.

known_risks: This diagnostic does not prove gameplay improvement. It should be used to guide future UNKNOWN_0 threshold changes, not to claim current agent strength.

next_candidates:
- Tune UNKNOWN_0 table threshold using split-aware holdout/stress diagnostics.
- Add deterministic local head-to-head evaluation seeds.
- Candidate deck selection review with split-aware reports.
