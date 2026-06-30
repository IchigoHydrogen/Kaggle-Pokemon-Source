# Pokemon Local Loop Record

## Version

version: `pokemon-20260624-v0-06d11`
baseline: `pokemon-20260624-v0-06d10`
canonical_notebook: `pokemon-20260624-v0-06d11.ipynb`
runtime_mode: `guarded_torch_policy`
promotion_type_target: `runtime_promote`

## Goal

Fix the v0-06d10 Kaggle validation crash without changing the policy hypothesis. v0-06d11 keeps the v0-06d10 guarded MAIN scorer design and changes only generated submission entrypoint compatibility plus archive-level loader validation.

## Root Cause

The v0-06d10 archive failed because `kaggle_environments.agent.get_last_callable` selects the last callable in raw `main.py`, not necessarily the global function named `agent`. In v0-06d10, the last callable was `_v06d10_features_batch`, so Kaggle called it as the agent with `(observation, configuration)`. The configuration dict was then iterated as options, producing string keys and the observed `'str' object has no attribute 'get'` crash.

A second loader-specific issue was found during v0-06d11 validation: the raw loader executes `main.py` without defining `__file__`. The guarded model path therefore needed a fallback based on the execution directory that Kaggle temporarily appends to `sys.path`.

## Implementation

- Created canonical notebook `pokemon-20260624-v0-06d11.ipynb`.
- Kept model architecture, feature definition, deck, thresholds, safe override types, hard veto types, and smoke protocol unchanged from v0-06d10.
- Re-ran fresh training under `/kaggle/working/pokemon-20260624-v0-06d11/`.
- Appended final `_kaggle_submission_entrypoint(obs_dict, configuration=None)` after all helper callables in generated `main.py`.
- Added archive-level validation using `kaggle_environments.agent.get_last_callable` on the actual archived `main.py`.
- Added loader-model path fallback for raw execution without `__file__`.
- Copied the downloaded failed validation episode files into the v0-06d11 output directory and used `81567646.json` as the entrypoint fixture.
- Kept generated archive flow inside the notebook; no manual repack was used as the promoted artifact.

## Execution

The canonical notebook executed end-to-end with exit code 0.

Key artifacts:
- `/kaggle/working/pokemon-20260624-v0-06d11.ipynb`
- `/kaggle/working/pokemon-20260624-v0-06d11-submission.tar.gz`
- `/kaggle/working/pokemon-20260624-v0-06d11/pokemon-20260624-v0-06d11-main_hybrid_report.json`
- `/kaggle/working/pokemon-20260624-v0-06d11/pokemon-20260624-v0-06d11-main_runtime_probe_report.json`
- `/kaggle/working/pokemon-20260624-v0-06d11/pokemon-20260624-v0-06d11-kaggle_loader_validation_report.json`
- `/kaggle/working/pokemon-20260624-v0-06d11/pokemon-20260624-v0-06d11-promotion-decision.json`

## Results

Fresh training:
- decisions: 101,253
- option rows: 1,278,625
- feature dimension: 96
- extraction time: 277.1s
- train time: 13.0s
- epochs_run: 37
- best_epoch: 31
- best_valid_top1: 0.5128

Holdout:
- model_top1: 0.5090
- hybrid_top1: 0.4507
- v0-06d10 hybrid baseline: 0.4466
- holdout_hybrid_minus_v06d10: +0.0040
- dangerous bucket hybrid delta: 0

Runtime probe:
- torch_load_ok: true
- runtime feature crosscheck: passed, max_err=0
- smoke games: 20
- illegal_actions: 0
- exceptions: 0
- action_changes: 352 / 643 decisions
- latency p50: 0.249 ms
- latency p95: 0.557 ms
- latency p99: 0.962 ms
- max latency: 4.831 ms

Kaggle loader validation:
- selected_callable_name: `_kaggle_submission_entrypoint`
- selected_callable_argcount: 2
- module_agent_name: `agent`
- loader_model_loaded: true
- fixture_source: `81567646.json`
- fixture one-arg call: returned list
- fixture two-arg call: returned list
- entrypoint_ok: true

Archive validation:
- contains `main.py`, `deck.csv`, `cg/api.py`, `cg/game.py`, `models/main_option_scorer.pt`
- no `__pycache__`, `.pyc`, `.pyo`, or `Zone.Identifier`
- `main.py` compiles

## Promotion Decision

decision: `runtime_promote`

reason: v0-06d11 fixes the Kaggle raw-loader entrypoint failure, validates model loading through the same loader path, passes the 81567646 fixture check, passes local guarded smoke safety, and keeps holdout quality at least baseline level. The change is scoped to submission entrypoint/runtime packaging compatibility and does not broaden policy behavior.

known_risks:
- The local `kaggle_environments` version is 1.27.0 while the failed episode reported 1.30.1. The verified behavior matches the observed traceback and loader contract, but exact hosted runtime remains the final arbiter.
- The fixture call at step 0 returns the deck list action used by the rule agent in that initial observation shape. This is accepted by the local loader check and is consistent with the no-crash goal, but hosted validation should still be watched.
- Torch availability in hosted Kaggle remains environment-dependent. If torch import fails, the generated agent falls back to rule behavior.

next_candidates:
- Submit `pokemon-20260624-v0-06d11-submission.tar.gz` for validation.
- If validation passes, use v0-06d11 as the next runtime baseline.
- If validation fails, inspect the new episode logs first before changing policy logic.
