# Pokemon Local Loop Record

## Version Goal

version: `pokemon-20260623-v0-05d2`

baseline: `pokemon-20260622-v0-05d1`

goal: Make loop artifacts reproducible and self-consistent before starting gameplay optimization.

hypothesis: If every generated artifact path recorded in the run summary matches the actual prefixed file on disk, then later loops can compare versions mechanically without path ambiguity or stale bare `submission.tar.gz` collisions.

change_scope: Artifact path recording and submission archive naming only.

datasets:
- `top200-20260622-ranking`
- `pokemon-tcg-ai-battle-episodes-2026-06-22`
- `pokemon-tcg-ai-battle` base data

success_criteria:
- Notebook executes fully.
- `RUN_PREFIX` is `pokemon-20260623-v0-05d2`.
- All output artifacts are under `/kaggle/working/pokemon-20260623-v0-05d2/`.
- Main output files use `pokemon-20260623-v0-05d2-` prefix.
- Submission archives are named `pokemon-20260623-v0-05d2-submission*.tar.gz`.
- Run summary artifact paths point to existing files.
- No `submission.tar.gz` is newly required for this version.
- Safety gates remain clean: no errors, no illegal actions, no Python cache in archives.

rollback_criteria:
- Notebook execution fails.
- Any required artifact path in run summary points to a missing file.
- Submission archive is missing required files.
- Safety gates regress compared with baseline.

expected_artifacts:
- `pokemon-20260623-v0-05d2-v05_run_summary.json`
- `pokemon-20260623-v0-05d2-v05_errors.json`
- `pokemon-20260623-v0-05d2-submission.tar.gz`
- `pokemon-20260623-v0-05d2-promotion-decision.json`

## Pre-Implementation Self-Review

experiment_plan: Create v0-05d2 as a reproducibility and artifact-consistency version. Do not alter strategy, deck selection, policy thresholds, model features, or evaluation game counts.

implementation_plan: Copy the v0-05d1 notebook to `pokemon-20260623-v0-05d2.ipynb`, update `EXPERIMENT_NAME` and `RUN_PREFIX`, keep prefixed submission archive naming, and fix JSON artifact path recording so `ARTIFACT_PATHS` stores the actual prefixed filenames returned by the artifact path helper.

why_this_is_the_next_best_step: The baseline already produced useful policy and model artifacts, but several JSON paths in `artifacts` point to unprefixed names while the files on disk are prefixed. If this is not fixed first, later version comparisons can silently read missing or stale artifacts.

what_would_make_this_result_untrustworthy: Treating this version as a gameplay improvement would be untrustworthy because it intentionally does not change gameplay. Any win-rate movement should be considered noise unless caused by execution nondeterminism.

expected_failure_modes:
- A path-recording change accidentally changes where files are written.
- Submission archive naming changes summary records but not the actual archive path.
- Existing direct `ARTIFACT_PATHS[...] = str(OUTPUT_DIR / ...)` assignments remain inconsistent.
- Notebook output still contains stale historical text from prior execution until re-run.

scope_guardrails:
- No deck list edits.
- No rule-agent logic edits.
- No UNKNOWN_0 threshold edits.
- No model architecture or feature edits.
- No Kaggle submission.

validation_plan:
- Run JSON validation before execution.
- Execute the full notebook in the `kaggle` container.
- Confirm the version output directory exists.
- Confirm run summary and errors JSON exist with the v0-05d2 prefix.
- Load run summary and check each absolute artifact path exists.
- Inspect `pokemon-20260623-v0-05d2-submission.tar.gz` for required files and absence of `__pycache__`, `.pyc`, and `.pyo`.
- Compare hard gates with baseline.

go_no_go: go. The plan is narrow, directly improves loop reliability, and does not mix gameplay hypotheses.

## Promotion Decision

decision: promote

reason: Artifact path recording and submission archive naming are now self-consistent. The full notebook executed successfully, all hard gates passed, and every absolute artifact path recorded in the run summary points to an existing file.

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
- smoke eval win rate: baseline `1.0`, v0-05d2 `1.0`
- smoke illegal actions: baseline `0`, v0-05d2 `0`
- confirm random same-deck win rate: baseline `1.0`, v0-05d2 `0.975`; not used as promotion evidence because this version makes no gameplay changes and local eval has stochastic variation.

known_risks:
- This version should not be claimed as a gameplay improvement.
- The current evaluation still lacks a deterministic holdout comparison harness.
- UNKNOWN_0 policy table size varied across executions, likely due training or threshold nondeterminism; future gameplay changes need fixed split and seed diagnostics.

next_candidates:
- UNKNOWN_0 policy activation tightening.
- Deterministic episode split and holdout comparison artifact.
- Candidate deck selection review with stronger local head-to-head evaluation.
