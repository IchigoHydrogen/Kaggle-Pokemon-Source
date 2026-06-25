# Pokemon Local Loop Record

## Version Goal

version: `pokemon-20260623-v0-06d1`

baseline: `pokemon-20260623-v0-05d6`

goal: Add a phase-aware tactical scoring pass for ordinary rule-agent decisions.

hypothesis: The current agent has a clean leakage-free UNKNOWN_0 validation baseline, but its largest remaining practical weakness is not UNKNOWN_0 coverage. A phase-aware tactical scorer for setup, benching, hand selection, switching, energy attachment, KO closing, and deckout avoidance should improve broad gameplay quality more than another narrow UNKNOWN_0 threshold tweak.

change_scope: Base rule-agent action scoring only, plus diagnostics needed to validate the change. Keep deck list, UNKNOWN_0 policy-table thresholds, MLP architecture, MLP features, and local evaluation game counts unchanged.

datasets:
- `top200-20260622-ranking`
- `pokemon-tcg-ai-battle-episodes-2026-06-22`
- `pokemon-tcg-ai-battle` base data

success_criteria:
- Notebook executes fully.
- `RUN_PREFIX` is `pokemon-20260623-v0-06d1`.
- Existing hard gates remain clean.
- Local evaluation has zero illegal actions and zero exceptions.
- Submission archives remain prefixed and valid.
- Tactical changes are visible in generated `main.py`.
- Confirm/meta local summaries do not show an obvious collapse versus v0-05d6.
- UNKNOWN_0 leakage-free diagnostics remain present.

rollback_criteria:
- Notebook execution fails.
- Any illegal action or exception appears in local evaluation.
- Submission archive checks fail.
- The final source imports torch or reads `deck.csv` at import time.
- Local confirm/meta performance collapses in a way not explained by variance.
- UNKNOWN_0 diagnostics or policy-table packaging are broken.

expected_artifacts:
- `pokemon-20260623-v0-06d1-v05_run_summary.json`
- `pokemon-20260623-v0-06d1-unknown0_policy_split_eval.parquet`
- `pokemon-20260623-v0-06d1-unknown0_policy_fit_split_summary.json`
- `pokemon-20260623-v0-06d1-submission.tar.gz`
- `pokemon-20260623-v0-06d1-promotion-decision.json`

## Pre-Implementation Self-Review

experiment_plan: Add one coherent tactical layer to the rule agent rather than tuning a single context. The layer should score valid options using explicit phase, KO, setup, resource, switch, and deckout signals, then let existing UNKNOWN_0 policy support remain in place as a fallback/assist.

implementation_plan: Copy v0-05d6 to `pokemon-20260623-v0-06d1.ipynb`, update `EXPERIMENT_NAME` and `RUN_PREFIX`, inspect the embedded `ALAKAZAM_RULE_AGENT_SOURCE`, and add small helper functions plus targeted score adjustments in existing selection functions. Prefer local helpers over wholesale rewrites, and keep the deck and model path unchanged.

why_this_is_the_next_best_step: v0-05d6 made holdout evaluation cleaner. The biggest remaining diagnosed gaps are broad tactical contexts such as `TO_HAND`, `SWITCH`, `TO_BENCH`, and deckout/endgame handling; improving these should affect many more live decisions than another UNKNOWN_0 table tweak.

what_would_make_this_result_untrustworthy: Replay agreement may go down even if local play improves, and local simple-agent win-rate is noisy. Conversely, local win-rate can improve by exploiting simple opponents without improving Kaggle strength. Promotion needs safety gates plus no obvious regression in confirm/meta and preserved diagnostics.

expected_failure_modes:
- Heuristics overfit to local simple opponents.
- Switch or bench preferences choose legal but strategically bad positions.
- Deckout avoidance blocks necessary draw/search actions.
- Tactical scoring conflicts with existing hard-coded special cases.
- The change touches too many branches and makes attribution difficult.

scope_guardrails:
- No deck list edits.
- No MLP architecture, feature, or threshold edits.
- No UNKNOWN_0 policy threshold edits.
- No evaluation game-count edits.
- No Kaggle submission.
- If the rule-agent source is too brittle for a broad pass, narrow to the highest-impact explicit scoring helpers instead of rewriting the agent.

validation_plan:
- Run JSON/code-cell validation.
- Execute smoke mode.
- Execute full notebook.
- Confirm hard gates and archive checks pass.
- Compare v0-06d1 against v0-05d6 on confirm/meta local summaries, UNKNOWN_0 holdout diagnostics, runtime policy stats, and errors.
- Record promotion decision as `promote`, `needs_followup`, or `reject`.

go_no_go: go. The plan is broad in coverage but narrow in mechanism: one phase-aware tactical scoring layer inside the existing rule agent, with no deck/model/UNKNOWN_0 threshold changes.

## Promotion Decision

decision: promote

promotion_type: tactical_policy_baseline

reason: Full notebook execution passed with clean hard gates. The phase-aware tactical scoring layer is present in the final submission archive and local confirm/meta summaries improved versus v0-05d6 without illegal actions or errors.

hard_gates:
- `RUN_MODE=full`
- `error_count=0`
- local smoke errors: `0`
- local smoke illegal actions: `0`
- local confirm errors: `0`
- local confirm illegal actions: `0`
- local meta errors: `0`
- local meta illegal actions: `0`
- submission archives are prefixed and contain `main.py`, `deck.csv`, `cg/api.py`, and `cg/game.py`
- archive inspection found no `__pycache__`, `.pyc`, `.pyo`, or torch-named files
- archive `main.py` contains `tactical_card_bonus`, `early_game`, and `op_prize_count`
- archive `main.py` does not import torch

baseline_comparison:
- v0-05d6 confirm: `73/80`, win rate `0.9125`, avg steps `93.225`
- v0-06d1 confirm: `76/80`, win rate `0.95`, avg steps `91.8`
- v0-05d6 meta: `70/80`, win rate `0.875`, avg steps `109.075`
- v0-06d1 meta: `76/80`, win rate `0.95`, avg steps `100.275`
- v0-05d6 smoke: `16/16`, win rate `1.0`
- v0-06d1 smoke: `15/16`, win rate `0.9375`
- v0-05d6 policy table entries: `50`
- v0-06d1 policy table entries: `52`
- v0-05d6 holdout policy-hit rate: `0.694245`
- v0-06d1 holdout policy-hit rate: `0.727418`
- v0-05d6 holdout policy correctness on hits: `0.602763`
- v0-06d1 holdout policy correctness on hits: `0.601099`
- v0-05d6 runtime UNKNOWN_0 policy selected: `8`
- v0-06d1 runtime UNKNOWN_0 policy selected: `3`

known_risks:
- This version may improve tactical intent without proving Kaggle strength. Local simple-agent eval is useful for safety and rough direction, not final strength proof.
- Smoke win-rate dipped from `1.0` to `0.9375`, though with no errors or illegal actions.
- Runtime UNKNOWN_0 policy selections decreased, so gains appear to come from base tactical scoring, not policy-table activity.
- The tactical pass changes several related scoring branches, so future work should add context-level ablations.

next_candidates:
- Deterministic baseline-vs-candidate head-to-head seeds.
- Context-specific imitation diagnostics for `TO_HAND`, `SWITCH`, and `TO_BENCH`.
- Candidate deck selection after tactical policy stabilizes.
