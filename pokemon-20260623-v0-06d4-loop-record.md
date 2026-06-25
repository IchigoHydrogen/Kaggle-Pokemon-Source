# Pokemon Local Loop Record

## Version Goal

version: `pokemon-20260623-v0-06d4`

baseline: `pokemon-20260623-v0-06d3` runtime, `pokemon-20260623-v0-06d1` gameplay

goal: Connect a trained PyTorch reranker to real action selection behind strict guards.

hypothesis: A 329k-parameter PyTorch option reranker trained to imitate the current v0-06d1 rule teacher can be packaged and used in production-form `main.py` without illegal actions, runtime issues, or obvious local durability collapse. This loop validates the action-changing neural path; it does not claim a large strength improvement.

change_scope: Add training-data collection from local self-play/random-opponent games, train a small option-level reranker, package the model, and enable guarded single-option overrides in selected low-risk contexts. Keep the deck and v0-06d1 rule policy otherwise unchanged.

datasets:
- local simulator states generated from v0-06d1 gameplay against random agents
- `pokemon-tcg-ai-battle` base data and `cg/` simulator

success_criteria:
- Script executes fully in the Kaggle Docker container.
- Prefixed archive `pokemon-20260623-v0-06d4-submission.tar.gz` is generated in `/kaggle/working`.
- Archive contains `main.py`, `deck.csv`, `cg/api.py`, `cg/game.py`, and the packaged `.pt` model.
- Packaged `main.py` imports torch, loads the model, and can alter actions only through the guarded reranker.
- Local durability has zero errors and zero illegal actions.
- Latency remains well below act-timeout risk.
- Training/eval artifacts and promotion decision are written under `/kaggle/working/pokemon-20260623-v0-06d4/`.

rollback_criteria:
- Any illegal action, exception, model-load failure, or archive packaging failure occurs.
- Guarded reranker causes obvious durability collapse versus v0-06d3/v0-06d1 local baseline.
- NN override rate is uncontrolled or cannot be measured.
- The run cannot distinguish runtime/action-path success from a pure shadow-mode artifact.

expected_artifacts:
- `pokemon-20260623-v0-06d4-submission.tar.gz`
- `pokemon-20260623-v0-06d4-training_report.json`
- `pokemon-20260623-v0-06d4-runtime_report.json`
- `pokemon-20260623-v0-06d4-promotion-decision.json`

## Pre-Implementation Self-Review

experiment_plan: Train an option-level PyTorch reranker from local v0-06d1 teacher decisions, then inject it into the existing rule agent so it may override exactly one selected option in guarded low-risk contexts when confidence and margin are high.

implementation_plan: Reuse v0-06d3 packaging/durability helpers, generate simulator observations with v0-06d1 as teacher, train a 64-feature 329k-parameter binary scorer, save the state dict, inject matching runtime feature extraction and guarded rerank code into baseline `main.py`, build the prefixed archive, and run durability/latency checks.

why_this_is_the_next_best_step: v0-06d3 proved official runtime feasibility for this parameter scale. The next missing piece is not a bigger model but a safe production path where NN output can actually affect actions while remaining measurable and reversible.

what_would_make_this_result_untrustworthy: A clean run with zero overrides would only prove shadow-like packaging, not action-changing NN safety. Conversely, many overrides with only random-agent wins could be local overfit. Teacher imitation accuracy can improve while gameplay remains unchanged or worse.

expected_failure_modes:
- Feature mismatch between training and packaged runtime.
- Model confidently overrides the teacher in rare contexts for bad reasons.
- Multi-select contexts create illegal or semantically poor choices, so this loop restricts to single-option overrides.
- Ability-use bookkeeping becomes stale if reranking happens after stateful updates, so reranking must occur before the existing ability flag block.
- Local random opponents fail to expose Kaggle-relevant mistakes.

scope_guardrails:
- No deck changes.
- No broad heuristic rewrite.
- No replay/holdout tuning in this loop.
- Only single-option guarded overrides.
- Keep fallback to baseline selected action whenever torch is unavailable, timing budget is low, confidence is low, context is disallowed, or shape checks fail.

validation_plan:
- Compile the generated script and generated `main.py`.
- Train/evaluate teacher imitation on deterministic game-id split.
- Inspect archive contents.
- Run local durability against the same random-opponent suite used for v0-06d3.
- Record torch policy stats, override count, latency, errors, illegal actions, and promotion decision.

go_no_go: go. The hypothesis is narrow enough: validate a trained, guarded, action-changing NN path without claiming final strength.

## Implementation Note

The first execution completed but produced `eligible=0` and `overrides=0` because the runtime `SelectContext` values are integer-like in the Kaggle `cg` environment rather than name-bearing enum objects. The guard was corrected to accept both context names and integer IDs:

- `SWITCH=3`
- `TO_ACTIVE=4`
- `TO_BENCH=5`
- `TO_HAND=7`
- `ATTACH_FROM=21`

The corrected run is the promoted v0-06d4 result below.

## Promotion Decision

decision: promote

promotion_type: guarded_torch_policy_path

reason: The trained PyTorch reranker was packaged into a prefixed submit archive, loaded successfully in generated `main.py`, changed actions through the guarded runtime path, and passed local durability with zero errors and zero illegal actions.

hard_gates:
- final archive: `/kaggle/working/pokemon-20260623-v0-06d4-submission.tar.gz`
- required archive files missing: `[]`
- archive has `__pycache__`: `False`
- archive has `.pyc/.pyo`: `False`
- model params: `329,089`
- training games: `96`
- training option rows: `45,189`
- training decisions: `5,845`
- valid option rows: `10,081`
- valid decision top-1 imitation: `0.5692068429`
- valid single-choice top-1 imitation: `0.5560936239`
- durability games: `96`
- durability wins: `89`
- durability win rate: `0.9270833333`
- durability errors: `0`
- durability illegal actions: `0`
- policy calls: `5,750`
- policy eligible calls: `907`
- policy same-top calls: `656`
- policy overrides: `133`
- policy inference calls: `907`
- options scored by torch: `4,428`
- act latency p50: `0.0001058250s`
- act latency p95: `0.0003054171s`
- act latency p99: `0.0004356888s`
- act latency max: `0.0013519880s`

baseline_comparison:
- v0-06d3 proved official runtime feasibility in a submitted Kaggle match that completed in about `2` minutes.
- v0-06d4 keeps the same approximate parameter scale and moves from shadow inference to guarded action-changing inference.
- v0-06d3 local durability: `84/96`, win rate `0.875`, avg steps `105.8125`, errors `0`, illegal actions `0`.
- v0-06d4 local durability: `89/96`, win rate `0.9270833333`, avg steps `109.5`, errors `0`, illegal actions `0`.
- Treat the local win-rate gain cautiously because this suite uses random opponents and is not a Kaggle-strength proof.

known_risks:
- The model is trained on self-generated v0-06d1 teacher decisions, so it mostly validates the neural action path rather than learning a stronger independent policy.
- Teacher-imitation valid top-1 is only about `57%`, so the current 64-feature representation is not sufficient for broad high-confidence policy replacement.
- The `133` overrides passed local safety, but they are not proven better than the teacher on strong opponents.
- The guard is intentionally narrow: single-option contexts only, with confidence and margin checks.

next_candidates:
- Train on replay/log action targets or outcome/value targets instead of only self-generated teacher labels.
- Add context-level override diagnostics so we can see exactly where the NN disagrees with the teacher.
- Improve features before widening the context allow-list: public card identity, option semantics, HP/energy/bench summaries, and phase flags should matter more than the current compact hash-like features.
- Run guard-threshold ablations while keeping a clean holdout/promotion split.

## Notebook Management Update

The v0-06d4 loop has been converted into a primary runnable notebook:

- notebook: `/kaggle/working/pokemon-20260623-v0-06d4.ipynb`
- companion script: `/kaggle/working/pokemon-20260623-v0-06d4-torch-guarded-reranker.py`

The notebook was executed end-to-end with:

```text
V06D4_TRAIN_GAMES=96
V06D4_TRAIN_EPOCHS=8
V06_TORCH_PROBE_GAMES=96
jupyter nbconvert --execute --inplace --to notebook pokemon-20260623-v0-06d4.ipynb --ExecutePreprocessor.timeout=900
```

Notebook execution regenerated the training artifacts, model, generated `main.py`, prefixed submission archive, durability report, and promotion decision under the v0-06d4 prefix.

Notebook execution result:

- final archive: `/kaggle/working/pokemon-20260623-v0-06d4-submission.tar.gz`
- required archive files missing: `[]`
- archive has `__pycache__`: `False`
- archive has `.pyc/.pyo`: `False`
- model params: `329,089`
- training games: `96`
- training option rows: `40,516`
- training decisions: `5,808`
- valid decision top-1 imitation: `0.5520231214`
- valid single-choice top-1 imitation: `0.5377711294`
- durability games: `96`
- durability wins: `91`
- durability win rate: `0.9479166667`
- durability errors: `0`
- durability illegal actions: `0`
- policy eligible calls: `967`
- policy overrides: `129`
- policy inference calls: `967`
- options scored by torch: `4,610`
- act latency p50: `0.0001080520s`
- act latency p95: `0.0003379490s`
- act latency p99: `0.0005029332s`
- act latency max: `0.0010294800s`

Management decision: use the notebook as the canonical v0-06d4 runnable artifact, matching the v0-06d1-and-earlier workflow. The `.py` file remains as a companion/exported script for easier diffing and command-line execution.

## Full-Pipeline Notebook Completion

The earlier v0-06d4 notebook was a fast development notebook and did not include the v0-06d1-and-earlier full replay/diagnostic pipeline. That was insufficient for canonical version management.

The canonical `pokemon-20260623-v0-06d4.ipynb` has now been rebuilt from the v0-06d1 full notebook lineage:

- It preserves replay mining, deck diagnostics, UNKNOWN/UNKNOWN_0 diagnostics, offline MLP diagnostics, split summaries, submission variant generation, and local/confirm/meta evaluation.
- It changes `RUN_PREFIX` to `pokemon-20260623-v0-06d4`.
- It appends the v0-06d4 guarded PyTorch reranker stage after the standard rule/UNKNOWN_0 submission build.
- It overwrites the final prefixed archive with the guarded torch-policy archive.
- It writes the final `v05_run_summary.json` with `submission_policy_mode = guarded_torch_policy` and `use_neural_for_submission = true`.

The full notebook was executed successfully end-to-end with:

```text
V05_RUN_PREFIX=pokemon-20260623-v0-06d4
V05_RUN_MODE=full
V06D4_TRAIN_GAMES=96
V06D4_TRAIN_EPOCHS=8
V06_TORCH_PROBE_GAMES=96
jupyter nbconvert --execute --inplace --to notebook pokemon-20260623-v0-06d4.ipynb --ExecutePreprocessor.timeout=3600
```

Full-pipeline summary:

- notebook: `/kaggle/working/pokemon-20260623-v0-06d4.ipynb`
- final archive: `/kaggle/working/pokemon-20260623-v0-06d4-submission.tar.gz`
- final summary: `/kaggle/working/pokemon-20260623-v0-06d4/pokemon-20260623-v0-06d4-v05_run_summary.json`
- experiment: `v0_06d4_full_pipeline_guarded_torch_reranker`
- run mode: `full`
- episode files: `5,256`
- episode index rows: `5,062`
- decision rows: `1,494,391`
- option rows: `8,352,193`
- error count: `0`
- final submission policy mode: `guarded_torch_policy`
- final archive imports torch: `True`
- final archive includes model: `pokemon-20260623-v0-06d4-torch_policy_state.pt`
- required archive files missing: `[]`
- archive has `__pycache__`: `False`
- archive has `.pyc/.pyo`: `False`

Final v0-06d4 torch-stage result from the full notebook:

- model params: `329,089`
- training games: `96`
- training option rows: `44,455`
- training decisions: `6,092`
- valid top-1 imitation: `0.5582137161`
- valid single-choice top-1 imitation: `0.5411471322`
- durability games: `96`
- durability win rate: `0.96875`
- durability errors: `0`
- durability illegal actions: `0`
- policy eligible calls: `983`
- policy overrides: `151`
- act latency p99: `0.0004461591s`
- act latency max: `0.0011771040s`

Management decision: this full-pipeline notebook supersedes the earlier fast v0-06d4 notebook as the canonical runnable artifact. The earlier `.py` companion can remain for quick development and diffing, but future version management should follow the full notebook lineage unless explicitly marked as a fast-dev/probe artifact.
