# Pokemon Local Loop Record

## Version Goal

version: `pokemon-20260623-v0-06d7`

base_notebook: `pokemon-20260623-v0-06d6.ipynb`

baseline_for_comparison: `pokemon-20260623-v0-06d6` diagnostic baseline, gameplay inherited from the v0-05d6 rule/UNKNOWN_0 line

goal: Build the first MAIN replay-supervised learning pipeline on the v0-06d6 base. Use real replay decisions, not rule-teacher labels, to train and evaluate a small PyTorch option scorer for `SelectContext.MAIN`. Do not enable the model in submission yet.

hypothesis: The replay observations proven usable in v0-06d6 contain enough signal for a small PyTorch option scorer to rank actual human MAIN choices above the existing v0-05d6 rule baseline in at least some clear holdout buckets. If the model cannot beat the rule baseline on holdout or only improves aggregate metrics while damaging dangerous buckets such as `ATTACK`/`END`, MAIN submission adoption should not proceed.

change_scope: Add a MAIN dataset extraction, feature extraction, offline PyTorch training, holdout evaluation, and diagnostics cell to a v0-06d6-derived notebook. No gameplay logic change, no torch submission adoption, no TO_BENCH/TO_DECK rule fix, and no v0-06d1 tactical branch.

datasets:
- `pokemon-tcg-ai-battle-episodes-2026-06-22` replay observations/actions
- deterministic episode split already produced by the v0-05d6/v0-06d6 pipeline
- `top200-20260622-ranking`
- v0-06d6 generated rule submission archive for the rule baseline and deck

success_criteria:
- Notebook executes fully in `RUN_MODE=full`.
- MAIN dataset is built from real `alakazam` actor replay decisions with episode-level split hygiene.
- Feature extraction does not inspect the target action label.
- Offline PyTorch model trains on train split and evaluates on valid/holdout.
- Holdout report compares model vs rule baseline vs random baseline using top-1, top-3, chosen-rank, and logloss.
- Bucket diagnostics are emitted for option type, matchup, rank bucket, winner/loser, and split.
- Artifacts are written under `/kaggle/working/pokemon-20260623-v0-06d7/` with the `pokemon-20260623-v0-06d7-` prefix.

rollback_criteria:
- Replay MAIN extraction has high agent exception or bad-shape rate.
- Dataset accidentally leaks target action into features.
- Model improves train but not holdout.
- Model only beats random and fails to beat the rule baseline on holdout.
- Dangerous buckets (`ATTACK`, `END`, `ABILITY`) regress badly versus rule baseline.
- Notebook or existing hard gates fail.

expected_artifacts:
- `pokemon-20260623-v0-06d7-main_dataset_report.json`
- `pokemon-20260623-v0-06d7-main_model_report.json`
- `pokemon-20260623-v0-06d7-main_eval_by_bucket.parquet`
- `pokemon-20260623-v0-06d7-main_holdout_predictions.parquet`
- `pokemon-20260623-v0-06d7-main_feature_spec.json`
- `pokemon-20260623-v0-06d7-main_option_scorer.pt`
- `pokemon-20260623-v0-06d7-promotion-decision.json`

## Pre-Implementation Self-Review

experiment_plan: Convert v0-06d6's faithful replay-observation machinery into a supervised learning dataset for `MAIN` only. Each decision becomes a variable-length option-ranking sample. Train a compact PyTorch scorer on real recorded actions and evaluate whether the model ranks recorded human choices above both random and the existing rule baseline on holdout.

implementation_plan: Copy/build from `pokemon-20260623-v0-06d6.ipynb`, bump `RUN_PREFIX` to `pokemon-20260623-v0-06d7`, append a MAIN supervised-learning cell before final summary, and add a small summary attachment to the final run summary. The cell loads the freshly built rule submission archive, walks replay episodes, filters `alakazam` actor `MAIN` decisions, extracts option features from raw obs/options, records rule predictions, trains a 300k-500k parameter PyTorch scorer, evaluates grouped decisions by split, writes reports, and records a diagnostic promotion decision.

why_this_is_the_next_best_step: v0-06d4 proved direct torch runtime feasibility but trained against the wrong teacher: the rule agent. v0-06d6 proved real replay observations/actions are faithful and highlighted MAIN as the high-volume long-term target. v0-06d7 must prove that real-data MAIN learning is measurable before any torch MAIN policy is adopted in submission.

what_would_make_this_result_untrustworthy:
- Using aggregate MAIN agreement alone. v0-06d6 showed aggregate rank/outcome agreement is weak and non-monotonic.
- Treating top-1 imitation as a strength proof. It is a supervised proxy, not a ladder proof.
- Letting feature extraction peek at the recorded action.
- Mixing TO_BENCH/TO_DECK fixes into the same version.
- Promoting torch submission adoption before dangerous MAIN sub-buckets are understood.

expected_failure_modes:
- The model learns option type/card priors but not stateful decision quality.
- Rare but critical actions such as `ATTACK` or `END` have unstable metrics.
- Winner/top200 weighting overfits because v0-06d6 showed weak aggregate gradients.
- Feature representation is still too shallow for MAIN planning.
- Holdout improvements are bucket-local but not broad enough for runtime adoption.

scope_guardrails:
- No submission torch adoption in v0-06d7.
- No rule-agent behavior change.
- No v0-06d1 tactical layer.
- No TO_BENCH/TO_DECK repair in this version.
- Use episode split, never row-random split.
- Treat this as a learning-pipeline and diagnostic version, not a gameplay promotion.

validation_plan:
- Smoke/full notebook execution.
- Dataset counts by split and label rate.
- Agent exception/bad-shape count during rule-baseline prediction.
- Holdout model-vs-rule-vs-random metrics overall and by option type.
- Promotion decision: `needs_followup` unless the model beats rule baseline broadly enough and dangerous buckets do not regress; even then, no submission adoption until a later guarded-runtime version.

go_no_go: go. The version has one hypothesis: whether real replay MAIN supervision can produce a measurable offline model improvement over the rule baseline without changing gameplay.

## Implementation & Clean Execution

- Built `pokemon-20260623-v0-06d7.ipynb` from `pokemon-20260623-v0-06d6.ipynb` with a new MAIN replay-supervised learning cell. The submitted gameplay archive remains the v0-05d6/v0-06d6-derived rule baseline; torch is trained offline only.
- Initial full execution produced `MAIN_LEARNING_REPORT.status=ok` but notebook-level `error_count=1` because the inherited v0-06d6 replay-agreement cell used a hardcoded temporary agent filename (`pokemon-20260623-v0-06d6-replay_agreement_agent.py`) and failed under the v0-06d7 output directory.
- Fixed `_build_v06d7_nb.py` so the inherited replay-agreement cell writes its temporary agent as `RUN_PREFIX + '-replay_agreement_agent.py'` using direct `Path.write_text(...)`. Rebuilt the notebook from the builder so the fix is reproducible.
- Smoke execution completed cleanly: `run_mode=smoke`, `error_count=0`, `v05_errors=[]`, replay agreement `status=ok`, MAIN learning `status=ok`.
- Full execution completed cleanly: `run_mode=full`, `error_count=0`, `v05_errors=[]`, replay agreement `status=ok`, MAIN learning `status=ok`.

## Full Results

Replay-agreement inherited diagnostic:
- episodes processed: 5278
- bad episodes skipped: 46
- eligible replay decisions: 488,455
- alakazam-actor decisions: 142,599

MAIN supervised dataset:
- decisions: 101,253
- option rows: 1,278,625
- feature dimension: 96
- model parameters: 345,473
- epochs: 5

Holdout summary:
- n: 14,847
- rule top-1: 0.1877
- model top-1: 0.5014
- model top-3: 0.7325
- random top-1 expected: 0.1281
- model minus rule top-1: +0.3137

Dangerous MAIN buckets:
- `ABILITY`: model improves over rule, +0.1418 top-1.
- `ATTACK`: model regresses versus rule, -0.1854 top-1.
- `END`: model regresses versus rule, -0.1544 top-1.

## Promotion Decision

decision: **needs_followup**

reason: The real replay-supervised MAIN PyTorch scorer is clearly learning useful signal and strongly beats the rule baseline on aggregate holdout imitation, but it regresses dangerous MAIN option types (`ATTACK`, `END`). Runtime adoption is therefore blocked in this version.

hard_gates:
- notebook full execution: yes
- `error_count=0`: yes
- replay-agreement inherited cell: fixed and passing
- MAIN learning stage: ok
- submission torch adoption: no
- submission archive prefix: `pokemon-20260623-v0-06d7-submission.tar.gz`
- submission `main.py` imports torch: no

next_candidates:
- Split MAIN policy adoption by safe option buckets first (`PLAY`/`ATTACH`/`EVOLVE`/`ABILITY`) instead of global MAIN override.
- Add hard veto/keep-rule handling for `ATTACK` and `END` before any runtime torch adoption.
- Improve features for attack/end timing before considering a full MAIN runtime policy.
- Keep this version as an offline learning-pipeline proof, not a gameplay promotion.
