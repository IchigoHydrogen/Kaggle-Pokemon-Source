# Pokemon Local Loop Record

## Version Goal

version: `pokemon-20260623-v0-06d8`

base_notebook: `pokemon-20260623-v0-06d7.ipynb` structure, but with fresh model training and no reuse of v0-06d7 model weights or predictions.

baseline_for_comparison: `pokemon-20260623-v0-06d7` offline MAIN learning result and the v0-05d6/v0-06d6-derived rule agent.

goal: Train a fresh replay-supervised PyTorch MAIN option scorer and evaluate whether it can be converted into a guarded hybrid policy that improves safe MAIN decisions without regressing dangerous actions.

hypothesis: A fresh-trained MAIN scorer can be useful if it is not allowed to control all MAIN actions. If the model only overrides rule decisions when its top option is in safe option buckets (`PLAY`, `ATTACH`, `EVOLVE`, `ABILITY`) and passes confidence/margin thresholds selected on valid, then holdout hybrid top-1 agreement should improve over the rule baseline while `ATTACK` and `END` remain non-regressed.

change_scope: Add guarded hybrid evaluation and threshold selection to the complete v0-06d7-style notebook. The model must be initialized and trained from scratch inside v0-06d8. No v0-06d7 `.pt` model, predictions, or threshold artifacts may be loaded. Submission runtime torch adoption is not promoted in this version.

datasets:
- `pokemon-tcg-ai-battle-episodes-2026-06-22` replay observations/actions
- deterministic episode split from the inherited v0-05d6/v0-06d6 pipeline
- fresh v0-06d8 submission archive for the rule baseline and deck

success_criteria:
- Notebook executes cleanly in smoke and full with `error_count=0`.
- MAIN model is fresh-trained in the notebook.
- Valid split is used to select guard thresholds; holdout is used only for final evaluation.
- Holdout report includes rule, model, and hybrid metrics.
- Hybrid evaluation records override rate, benefit (`rule wrong -> hybrid correct`), harm (`rule correct -> hybrid wrong`), and safe/dangerous bucket results.
- Runtime adoption remains blocked unless the hybrid clearly improves holdout while preserving `ATTACK` and `END`.
- Artifacts are written under `/kaggle/working/pokemon-20260623-v0-06d8/` with the `pokemon-20260623-v0-06d8-` prefix.

rollback_criteria:
- The model accidentally loads v0-06d7 weights/predictions.
- Valid threshold selection is performed on holdout.
- Hybrid improves aggregate top-1 but harms `ATTACK`/`END`.
- Override rate is too small to matter or harm is comparable to benefit.
- Notebook hard gates fail.

expected_artifacts:
- `pokemon-20260623-v0-06d8-main_dataset_report.json`
- `pokemon-20260623-v0-06d8-main_model_report.json`
- `pokemon-20260623-v0-06d8-main_hybrid_report.json`
- `pokemon-20260623-v0-06d8-main_eval_by_bucket.parquet`
- `pokemon-20260623-v0-06d8-main_hybrid_threshold_grid.parquet`
- `pokemon-20260623-v0-06d8-main_holdout_predictions.parquet`
- `pokemon-20260623-v0-06d8-main_option_scorer.pt`
- `pokemon-20260623-v0-06d8-promotion-decision.json`

## Pre-Implementation Self-Review

experiment_plan: Rebuild the MAIN supervised dataset from replay observations, train a compact PyTorch scorer from a fresh initialization, score all options, select guarded hybrid thresholds on valid, and evaluate the selected policy once on holdout.

implementation_plan: Copy the v0-06d7 builder/cell structure, bump `RUN_PREFIX` and experiment name, patch the inherited replay-agreement temp path, and extend the MAIN learning cell with valid-only threshold search. The hybrid policy defaults to the rule action and only overrides when the model top option is in a safe set and passes probability/margin thresholds. `ATTACK` and `END` are never model-selected override targets in v0-06d8.

why_this_is_the_next_best_step: v0-06d7 proved real replay MAIN supervision learns strong aggregate signal but breaks dangerous action types if used globally. The next question is not whether the model can learn; it is whether the learned signal can be used safely as a partial policy.

what_would_make_this_result_untrustworthy:
- Reusing v0-06d7 weights or predictions.
- Selecting thresholds on holdout.
- Reporting only model top-1 without benefit/harm accounting.
- Treating exact replay imitation as ladder strength.
- Ignoring dangerous option types.

expected_failure_modes:
- The best valid threshold overfits and does not transfer to holdout.
- Safe-bucket override still indirectly harms important turn sequences.
- The confidence distribution is poorly calibrated, making thresholds brittle.
- Aggregate improvement comes mostly from easy `PLAY` imitation while strategically important decisions remain weak.

scope_guardrails:
- Fresh train only.
- No full MAIN runtime adoption.
- No model override for `ATTACK`/`END`.
- No TO_BENCH/TO_DECK fixes.
- No v0-06d1 tactical branch.
- No holdout tuning.

validation_plan:
- Smoke notebook execution with reduced decisions/epochs.
- Full notebook execution with all available MAIN decisions and 5 epochs by default.
- Verify `error_count=0`, replay-agreement inherited cell ok, MAIN learning ok.
- Compare rule/model/hybrid on valid and holdout.
- Promotion decision: `needs_followup` unless hybrid is both useful and safe; even then, runtime adoption should be a later candidate with durability/latency checks.

go_no_go: go. The version has one hypothesis: a fresh MAIN model can become useful only through a valid-selected guarded hybrid policy, not through direct full-MAIN replacement.

## Implementation & Execution

- Built `pokemon-20260623-v0-06d8.ipynb` from the v0-06d6 complete pipeline, not from v0-06d7 artifacts.
- Added `_v06d8_main_hybrid_cell.py`, which rebuilds the MAIN replay dataset, trains a fresh PyTorch scorer, selects guard thresholds on valid, and evaluates holdout hybrid behavior.
- No v0-06d7 `.pt` weights or prediction artifacts are loaded. The report explicitly records `fresh_training=true` and `reuses_v0_06d7_weights_or_predictions=false`.
- Smoke run completed cleanly with `error_count=0`.
- First full run completed cleanly and showed strong aggregate hybrid gain, but `chosen_option_type=ABILITY` regressed. This violated the "do not break dangerous/important MAIN buckets" intent.
- Updated the guard: safe override types are now `PLAY,ATTACH,EVOLVE`; rule-veto/non-regression monitored types are `ATTACK,END,ABILITY`. Rebuilt, reran smoke, then reran full.
- Final full run completed cleanly: `run_mode=full`, `error_count=0`, `v05_errors=[]`.

## Final Full Results

Dataset / model:
- MAIN decisions: 101,253
- option rows: 1,278,625
- feature dimension: 96
- model parameters: 345,473
- epochs: 5
- model path: `pokemon-20260623-v0-06d8-main_option_scorer.pt`

Raw model holdout:
- rule top-1: 0.1877
- model top-1: 0.5012
- model top-3: 0.7340
- model minus rule top-1: +0.3135

Selected guard (chosen on valid):
- safe override types: `PLAY,ATTACH,EVOLVE`
- rule veto types: `ATTACK,END,ABILITY`
- probability threshold: 0.35
- margin threshold: 0.00

Holdout hybrid:
- hybrid top-1: 0.4457
- hybrid minus rule top-1: +0.2580
- override rate: 0.6325
- benefit count (`rule wrong -> hybrid correct`): 4,573
- harm count (`rule correct -> hybrid wrong`): 742
- benefit minus harm: +3,831
- dangerous/important bucket delta (`ATTACK/END/ABILITY` combined): 0.0000

Chosen-option bucket holdout:
- `PLAY`: 0.1938 -> 0.5318 (+0.3380)
- `ATTACH`: 0.0870 -> 0.3757 (+0.2887)
- `EVOLVE`: 0.2714 -> 0.5116 (+0.2402)
- `ABILITY`: 0.1032 -> 0.1032 (+0.0000)
- `ATTACK`: 0.5038 -> 0.5038 (+0.0000)
- `END`: 0.1544 -> 0.1544 (+0.0000)
- `RETREAT`: 0.0109 -> 0.0054 (-0.0054, small n=368; still a guard concern)

## Promotion Decision

decision: **needs_followup**

reason: The fresh-trained MAIN model plus guarded hybrid policy gives a large holdout imitation improvement while preserving `ABILITY/ATTACK/END`, so a guarded runtime probe can be considered next. It is not promoted directly because runtime torch adoption, latency/durability, and the small `RETREAT` regression have not been addressed.

hard_gates:
- notebook full execution: yes
- `error_count=0`: yes
- fresh training: yes
- v0-06d7 model/prediction reuse: no
- submission torch adoption: no
- submission archive prefix: `pokemon-20260623-v0-06d8-submission.tar.gz`
- submission `main.py` imports torch: no

artifacts:
- `pokemon-20260623-v0-06d8-main_hybrid_report.json`
- `pokemon-20260623-v0-06d8-main_model_report.json`
- `pokemon-20260623-v0-06d8-main_option_scorer.pt`
- `pokemon-20260623-v0-06d8-main_hybrid_threshold_grid.parquet`
- `pokemon-20260623-v0-06d8-main_hybrid_eval_by_bucket.parquet`
- `pokemon-20260623-v0-06d8-main_holdout_predictions.parquet`
- `pokemon-20260623-v0-06d8-promotion-decision.json`
- `pokemon-20260623-v0-06d8.ipynb`

next_candidates:
- v0-06d9 guarded-runtime probe: embed the fresh MAIN scorer and allow only `PLAY/ATTACH/EVOLVE` overrides with hard vetoes for `ATTACK/END/ABILITY/RETREAT`.
- Add runtime latency and durability evaluation before any Kaggle-facing submission.
- Consider a calibration/threshold refinement only on valid, with holdout untouched.
