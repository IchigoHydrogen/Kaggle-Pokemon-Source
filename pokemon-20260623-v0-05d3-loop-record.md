# Pokemon Local Loop Record

## Version Goal

version: `pokemon-20260623-v0-05d3`

baseline: `pokemon-20260623-v0-05d2`

goal: Add deterministic episode split and holdout/stress summary artifacts before gameplay optimization.

hypothesis: A stable episode-level `train` / `valid` / `holdout` split plus stress-slice summaries will make future local loop decisions less vulnerable to tuning on visible metrics or accidental train/valid leakage.

change_scope: Reporting and validation harness only. Do not change deck choice, rule-agent behavior, UNKNOWN_0 thresholds, model features, model training inputs, or local evaluation game counts.

datasets:
- `top200-20260622-ranking`
- `pokemon-tcg-ai-battle-episodes-2026-06-22`
- `pokemon-tcg-ai-battle` base data

success_criteria:
- Notebook executes fully.
- `RUN_PREFIX` is `pokemon-20260623-v0-05d3`.
- Deterministic split artifacts are generated under `/kaggle/working/pokemon-20260623-v0-05d3/`.
- Split assignment is episode-level and stable from `episode_id`.
- Split ratios are approximately train 70%, valid 15%, holdout 15%.
- Stress summary includes UNKNOWN_0, top200/top-tier, long-game, losing-decision, and matchup/tier slices where available.
- Run summary records split artifact paths and row counts.
- Existing hard gates remain clean.

rollback_criteria:
- Notebook execution fails.
- Split artifacts are missing or have zero rows in full mode.
- Any split assignment is not deterministic by `episode_id`.
- Any required artifact path in run summary points to a missing file.
- Safety gates regress compared with v0-05d2.

expected_artifacts:
- `pokemon-20260623-v0-05d3-episode_split.parquet`
- `pokemon-20260623-v0-05d3-split_summary.parquet`
- `pokemon-20260623-v0-05d3-stress_slice_summary.parquet`
- `pokemon-20260623-v0-05d3-v05_run_summary.json`
- `pokemon-20260623-v0-05d3-promotion-decision.json`

## Pre-Implementation Self-Review

experiment_plan: Add fixed split and stress reporting so later behavior changes can be judged against a reproducible holdout. This version is an evaluation-harness improvement, not a gameplay experiment.

implementation_plan: Copy v0-05d2 to `pokemon-20260623-v0-05d3.ipynb`, update `EXPERIMENT_NAME` and `RUN_PREFIX`, add stable hash helpers, build `EPISODE_SPLIT_DF`, `SPLIT_SUMMARY_DF`, and `STRESS_SLICE_SUMMARY_DF` after replay mining, save them with prefix, and include counts/paths in `V05_RUN_SUMMARY`.

why_this_is_the_next_best_step: v0-05d2 fixed artifact consistency. The next risk is local overfitting: current reports have model valid splits but no reusable episode-level holdout/stress artifact for loop promotion decisions.

what_would_make_this_result_untrustworthy: Claiming gameplay improvement from this version would be untrustworthy. Also, split metrics would be weak if the split were row-level instead of episode-level, if hash buckets were unstable across runs, or if stress slices were defined using future outcome in a way later training uses.

expected_failure_modes:
- Split assignment accidentally changes downstream training data.
- Stress summary is too broad to catch UNKNOWN_0 regressions.
- Summary paths drift from actual prefixed files.
- Small smoke-mode data creates sparse split/stress tables; full mode must be used for promotion.

scope_guardrails:
- No deck list edits.
- No rule-agent logic edits.
- No UNKNOWN_0 threshold edits.
- No model feature or training split edits.
- No Kaggle submission.

validation_plan:
- Run JSON validation before execution.
- Execute smoke mode to catch syntax/runtime issues.
- Execute full notebook in the `kaggle` container.
- Confirm split artifacts exist and have nonzero rows.
- Confirm split ratios are close to 70/15/15 in full mode.
- Confirm run summary artifact paths exist.
- Confirm standard submission hard gates still pass.

go_no_go: go. The plan is narrow, tests a harness capability directly, and prepares the next gameplay loop without mixing in behavior changes.

## Promotion Decision

decision: promote

reason: Deterministic episode-level train/valid/holdout split and stress-slice summary artifacts were added without changing gameplay behavior. The full notebook executed successfully, all hard gates passed, split artifacts were nonempty, and all recorded artifact paths exist.

hard_gates:
- full notebook execution: pass
- run mode: full
- error count: 0
- errors JSON length: 0
- local eval status: ok
- smoke illegal action count: 0
- missing artifact paths in run summary: 0
- unprefixed output files: 0
- prefixed submission archives: pass
- submission required files present: pass
- submission Python cache files: 0

baseline_comparison:
- selected deck hash: unchanged, `e702674c1864`
- final submission variant: unchanged, `unknown0_policy_table`
- decision rows: unchanged, `1,494,391`
- option rows: unchanged, `8,352,193`
- smoke eval win rate: baseline `1.0`, v0-05d3 `1.0`
- smoke illegal actions: baseline `0`, v0-05d3 `0`
- confirm random same-deck win rate: baseline `0.975`, v0-05d3 `0.9`; not used as promotion evidence because this version makes no gameplay changes and local eval has stochastic variation.
- confirm first-valid same-deck win rate: baseline `0.975`, v0-05d3 `1.0`

split_artifacts:
- episode split rows: `5,062`
- split summary rows: `3`
- stress slice summary rows: `60`
- stress slice count: `20`
- train episodes: `3,549` (`70.11%`)
- valid episodes: `732` (`14.46%`)
- holdout episodes: `781` (`15.43%`)
- holdout decision rows: `228,824`
- holdout UNKNOWN_0 decisions: `122,283`

known_risks:
- This version should not be claimed as a gameplay improvement.
- The split/stress artifacts are reporting-only; model training still uses existing internal train/valid splits.
- Local head-to-head evaluation seeds are still not fixed strongly enough for gameplay promotion decisions.

next_candidates:
- UNKNOWN_0 policy activation tightening using the new holdout/stress artifacts.
- Candidate deck selection review with split-aware reports.
- Deterministic head-to-head local evaluation seeds.
