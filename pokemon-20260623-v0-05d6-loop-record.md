# Pokemon Local Loop Record

## Version Goal

version: `pokemon-20260623-v0-05d6`

baseline: `pokemon-20260623-v0-05d5`

goal: Build the UNKNOWN_0 policy table without deterministic holdout leakage.

hypothesis: Excluding deterministic `holdout` episodes from UNKNOWN_0 policy-table distillation will produce a cleaner promotion signal. If the table still has useful holdout coverage and no safety regressions, future threshold changes can use holdout as a real promotion check instead of a partially leaked diagnostic.

change_scope: UNKNOWN_0 policy-table distillation input and diagnostics only. Do not change deck choice, base rule-agent behavior, MLP architecture, MLP features, threshold constants, or local evaluation game counts.

datasets:
- `top200-20260622-ranking`
- `pokemon-tcg-ai-battle-episodes-2026-06-22`
- `pokemon-tcg-ai-battle` base data

success_criteria:
- Notebook executes fully.
- `RUN_PREFIX` is `pokemon-20260623-v0-05d6`.
- Existing hard gates remain clean.
- Policy table construction reports deterministic split filtering.
- Holdout rows are excluded from table fitting and still present in evaluation.
- Holdout UNKNOWN_0 policy-hit decisions remain nonzero.
- Holdout policy correctness on hits is recorded and does not reveal an obvious collapse versus v0-05d5.
- No gameplay behavior changes are made outside the resulting distilled UNKNOWN_0 policy table.

rollback_criteria:
- Notebook execution fails.
- Evaluation artifacts are missing or empty in full mode.
- Holdout split has zero UNKNOWN_0 policy-evaluable decisions.
- Holdout policy-hit decisions collapse to zero.
- Any hard gate fails.
- Runtime policy table becomes empty in full mode unless diagnostics clearly justify fallback.
- Any required artifact path in run summary points to a missing file.

expected_artifacts:
- `pokemon-20260623-v0-05d6-unknown0_policy_split_eval.parquet`
- `pokemon-20260623-v0-05d6-unknown0_policy_stress_eval.parquet`
- `pokemon-20260623-v0-05d6-unknown0_policy_fit_split_summary.json`
- `pokemon-20260623-v0-05d6-v05_run_summary.json`
- `pokemon-20260623-v0-05d6-promotion-decision.json`

## Pre-Implementation Self-Review

experiment_plan: Use v0-05d5 as the baseline and change only the rows used for UNKNOWN_0 policy-table distillation. The MLP will still produce its validation prediction frame, but the table builder will receive only rows whose deterministic episode split is not `holdout`.

implementation_plan: Copy v0-05d5 to `pokemon-20260623-v0-05d6.ipynb`, update `EXPERIMENT_NAME` and `RUN_PREFIX`, add a helper that joins `pred_df` to `EPISODE_SPLIT_DF`, creates `UNKNOWN0_POLICY_FIT_PRED_DF` excluding `holdout`, saves a fit split summary JSON, builds the policy table from that filtered frame, and evaluates the resulting table on the full prediction frame by split using the existing v0-05d5 diagnostics.

why_this_is_the_next_best_step: v0-05d5 made holdout visibility possible but did not prevent the current table from being fitted with rows later labeled as holdout. Fixing that leakage comes before using holdout to tune UNKNOWN_0 thresholds.

what_would_make_this_result_untrustworthy: Treating local smoke win-rate noise as proof of improvement would be untrustworthy. A smaller table can look safer by firing less often, so promotion needs holdout coverage, correctness, and runtime hit counts in addition to hard gates.

expected_failure_modes:
- The filtered fit set may be too small and produce a sparse or empty policy table.
- Holdout correctness may drop once leaked rows are excluded.
- Runtime policy hits may become too rare to matter.
- The join may silently label rows as `unknown` if episode IDs do not align.

scope_guardrails:
- No deck list edits.
- No base rule-agent edits.
- No UNKNOWN_0 threshold edits.
- No model feature or architecture edits.
- No local evaluation game-count edits.
- No Kaggle submission.

validation_plan:
- Run JSON/code-cell validation.
- Execute smoke mode.
- Execute full notebook.
- Confirm fit split summary shows `holdout` excluded from table fitting.
- Confirm split-aware evaluation artifacts are nonempty and include `holdout`.
- Compare holdout policy-hit/correctness against v0-05d5.
- Confirm hard gates and submission archive checks still pass.

go_no_go: go. The hypothesis is narrow, fixes a known validation weakness, and keeps holdout as an evaluation signal rather than a fitting input.

## Promotion Decision

decision: promote

promotion_type: leakage_free_diagnostic_baseline

reason: Full notebook execution passed with clean hard gates. UNKNOWN_0 policy-table fitting now excludes deterministic holdout decisions, and holdout evaluation remains nonzero with comparable correctness on hits versus v0-05d5.

hard_gates:
- `RUN_MODE=full`
- `error_count=0`
- `unknown0_policy_table_entries=50`
- fit split filter status: `ok`
- excluded split: `holdout`
- excluded holdout fit decisions: `2502`
- `unknown0_policy_decision_eval_rows=16446`
- `unknown0_policy_split_eval_rows=3`
- `unknown0_policy_stress_eval_rows=12`
- holdout UNKNOWN_0 policy-evaluable decisions: `2502`
- holdout policy-hit decisions: `1737`
- holdout policy-hit rate: `0.694245`
- holdout policy correctness on hits: `0.602763`
- submission archives are prefixed and contain `main.py`, `deck.csv`, `cg/api.py`, and `cg/game.py`
- archive inspection found no `__pycache__`, `.pyc`, `.pyo`, or torch-named files

baseline_comparison:
- v0-05d5 policy table entries: `59`
- v0-05d6 policy table entries: `50`
- v0-05d5 holdout policy-hit rate: `0.777378`
- v0-05d6 holdout policy-hit rate: `0.694245`
- v0-05d5 holdout policy correctness on hits: `0.601542`
- v0-05d6 holdout policy correctness on hits: `0.602763`
- v0-05d5 runtime selected actions in variant smoke: `6`
- v0-05d6 runtime selected actions in variant smoke: `8`
- v0-05d6 local confirm summary: `73/80`, win rate `0.9125`, errors `0`, illegal actions `0`

known_risks: This version may reduce table coverage without improving runtime strength. It should be promoted only as a cleaner validation baseline unless local evidence also supports strength improvement.

next_candidates:
- Tune UNKNOWN_0 table threshold using the leakage-free split evaluation.
- Add deterministic local head-to-head evaluation seeds.
- Candidate deck selection review with split-aware reports.
