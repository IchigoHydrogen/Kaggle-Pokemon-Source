# Pokemon Local Loop Record

## Version Goal

version: `pokemon-20260623-v0-05d4`

baseline: `pokemon-20260623-v0-05d3`

goal: Reduce UNKNOWN_0 policy-table misfire risk by pruning low-accuracy table entries.

hypothesis: The UNKNOWN_0 distilled policy table currently includes some bucket/fallback entries with offline `top_correct_rate` below 0.75. Requiring a stricter table-entry accuracy floor should remove the riskiest policy overrides while preserving high-confidence coverage.

change_scope: UNKNOWN_0 policy-table construction only. Do not change deck choice, base rule-agent heuristics, model architecture, model features, local evaluation game counts, or deterministic split generation.

datasets:
- `top200-20260622-ranking`
- `pokemon-tcg-ai-battle-episodes-2026-06-22`
- `pokemon-tcg-ai-battle` base data

success_criteria:
- Notebook executes fully.
- `RUN_PREFIX` is `pokemon-20260623-v0-05d4`.
- Existing hard gates remain clean.
- UNKNOWN_0 policy table has fewer entries than v0-05d3.
- Minimum policy-table `top_correct_rate` is at least `0.75`.
- Holdout/stress split artifacts are still generated.
- Local smoke has zero illegal actions.
- Submission archive remains torch-free and cache-free.

rollback_criteria:
- Notebook execution fails.
- Policy table becomes empty.
- Any hard gate fails.
- Local eval produces illegal actions or errors.
- The change removes too much UNKNOWN_0 coverage without a clear safety benefit.

expected_artifacts:
- `pokemon-20260623-v0-05d4-unknown0_policy_table.parquet`
- `pokemon-20260623-v0-05d4-unknown0_policy_summary.json`
- `pokemon-20260623-v0-05d4-stress_slice_summary.parquet`
- `pokemon-20260623-v0-05d4-v05_run_summary.json`
- `pokemon-20260623-v0-05d4-promotion-decision.json`

## Pre-Implementation Self-Review

experiment_plan: Add a separate `UNKNOWN0_POLICY_MIN_TABLE_ACC` threshold with default `0.75` and use it only when admitting entries into the distilled UNKNOWN_0 policy table. Keep signature selection thresholds unchanged.

implementation_plan: Copy v0-05d3 to `pokemon-20260623-v0-05d4.ipynb`, update `EXPERIMENT_NAME` and `RUN_PREFIX`, add `UNKNOWN0_POLICY_MIN_TABLE_ACC`, replace the two table-entry `acc < UNKNOWN0_POLICY_MIN_HIGH_CONF_ACC` checks with `acc < UNKNOWN0_POLICY_MIN_TABLE_ACC`, and record the new threshold in `UNKNOWN0_POLICY_SUMMARY`.

why_this_is_the_next_best_step: v0-05d3 created holdout/stress artifacts. The most obvious low-risk gameplay change is to reduce policy-table override risk without changing the underlying model or rule fallback.

what_would_make_this_result_untrustworthy: Treating local head-to-head win-rate movement as definitive would be untrustworthy because evaluation seeds are still noisy. Promotion should primarily rely on hard gates, table-pruning evidence, holdout/stress artifacts remaining intact, and absence of safety regressions.

expected_failure_modes:
- Table coverage drops too far, making UNKNOWN_0 policy mostly inert.
- Signature-level fallback rows are pruned more than intended.
- Runtime policy hit rate falls to zero.
- The stricter policy changes stochastic local evaluation but not in a reliably interpretable way.

scope_guardrails:
- No deck list edits.
- No base rule-agent heuristic edits.
- No UNKNOWN_0 model retraining changes.
- No model feature edits.
- No local evaluation game-count changes.
- No Kaggle submission.

validation_plan:
- Run JSON and code-cell validation before execution.
- Execute smoke mode.
- Execute full notebook in the `kaggle` container.
- Confirm `unknown0_policy_table` has nonzero entries and min `top_correct_rate >= 0.75`.
- Confirm policy table entry count is lower than v0-05d3.
- Confirm split/stress artifacts and summary paths exist.
- Confirm submission archives pass required-file/cache checks.
- Compare safety metrics with v0-05d3.

go_no_go: go. The plan is narrow, directly tests a single policy-pruning hypothesis, and keeps diagnostics sufficient without mixing in other gameplay changes.

## Promotion Decision

decision: promote

reason: UNKNOWN_0 policy-table low-accuracy entries were pruned while hard safety gates remained clean. This is promoted as a conservative safety-pruning baseline, not as proven win-rate improvement.

hard_gates:
- full notebook execution: pass
- run mode: full
- error count: 0
- errors JSON length: 0
- local eval status: ok
- smoke illegal action count: 0
- missing artifact paths in run summary: 0
- prefixed submission archives: pass
- submission required files present: pass
- submission Python cache files: 0

baseline_comparison:
- selected deck hash: unchanged, `e702674c1864`
- final submission variant: unchanged, `unknown0_policy_table`
- decision rows: unchanged, `1,494,391`
- option rows: unchanged, `8,352,193`
- UNKNOWN_0 policy entries: baseline `63`, v0-05d4 `62`
- UNKNOWN_0 policy table decisions: baseline `15,971`, v0-05d4 `15,583`
- minimum policy table `top_correct_rate`: baseline `0.660714`, v0-05d4 `0.75`
- mean policy table `top_correct_rate`: v0-05d4 `0.858826`
- runtime UNKNOWN_0 policy selected count in variant smoke: baseline `2`, v0-05d4 `1`
- smoke eval win rate: baseline `1.0`, v0-05d4 `0.9375`
- confirm random same-deck win rate: baseline `0.9`, v0-05d4 `1.0`
- confirm first-valid same-deck win rate: baseline `1.0`, v0-05d4 `0.925`
- local eval win-rate movement is not used as decisive promotion evidence because local eval is stochastic.

known_risks:
- This version is not proven to improve win rate.
- Runtime UNKNOWN_0 policy hit rate in local smoke remains very low.
- Split-aware UNKNOWN_0 holdout evaluation is still needed before more aggressive policy changes.
- Local head-to-head evaluation seeds remain noisy.

next_candidates:
- Add split-aware UNKNOWN_0 policy evaluation on holdout decisions.
- Add deterministic head-to-head local evaluation seeds.
- Candidate deck selection review with split-aware reports.
