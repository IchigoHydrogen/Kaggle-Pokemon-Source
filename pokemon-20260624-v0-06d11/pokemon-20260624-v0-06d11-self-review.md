# v0-06d11 Pre-Implementation Self-Review

experiment_plan: Run a fresh v0-06d11 notebook that reproduces the v0-06d10 guarded MAIN scorer pipeline, then changes only submission entrypoint generation and archive validation. The central test is whether the archived raw main.py is loaded by kaggle_environments as the intended final entrypoint instead of a helper callable.

implementation_plan: Copy/bump the v0-06d10 notebook to `pokemon-20260624-v0-06d11.ipynb`, update RUN_PREFIX and labels, update the guarded runtime cell to append a final `_kaggle_submission_entrypoint` callable after all helper definitions, and add archive-level checks using `get_last_callable`. The notebook will train a fresh model, rebuild the guarded archive, run the existing feature crosscheck and local smoke eval, and record loader/fixture validation in `main_runtime_probe_report.json` and `promotion-decision.json`.

why_this_is_the_next_best_step: The failed validation traceback shows `_v06d10_features_batch` was called directly by Kaggle as the agent. Local reproduction with `get_last_callable` confirms the archived v0-06d10 main.py selects `_v06d10_features_batch`, while `module.agent` itself is correct. Therefore the next step should target the loader contract, not policy learning or deck changes.

what_would_make_this_result_untrustworthy: If validation uses a different loader than `kaggle_environments.agent.get_last_callable`, if the notebook silently reuses v0-06d10 weights or reports, if the final callable check is run against an in-memory string but not the actual archive, if the fixture check bypasses the archive, or if a policy change is introduced at the same time.

expected_failure_modes:
- The final entrypoint is not actually the last callable after code generation.
- The final entrypoint has the wrong arity for Kaggle's one-argument/trimmed call path.
- The archive-level check passes locally but misses a Python 3.11/Kaggle 1.30.1 difference.
- The fixture observation has `current=None` and exposes an import-time or rule-fallback bug.
- Torch load succeeds locally but fails in Kaggle, requiring the rule-only fallback path.

scope_guardrails:
- No deck change.
- No model architecture change.
- No feature definition change except defensive type handling if required for the loader fixture.
- No threshold change beyond fresh valid selection from the current run.
- No new action families or override contexts.
- No hand-edited repack of v0-06d10 archive as the promoted artifact.
- Notebook remains the canonical runnable artifact; helper scripts are not the version artifact.

validation_plan:
- Execute `pokemon-20260624-v0-06d11.ipynb` end-to-end.
- Confirm artifacts are written under `/kaggle/working/pokemon-20260624-v0-06d11/`.
- Confirm submission archive is `/kaggle/working/pokemon-20260624-v0-06d11-submission.tar.gz`.
- Extract archived main.py and run `get_last_callable(raw, path=extracted/main.py)`.
- Assert selected callable name is `_kaggle_submission_entrypoint` and not `_v06d*_features_batch`.
- Call the selected callable with the downloaded 81567646 step-0 observation and with observation/configuration if the callable accepts two arguments.
- Run existing local guarded smoke eval and safety gates.
- Compare holdout model/hybrid top-1 to v0-06d10 as a regression check only.

promotion_evidence_required: Notebook full execution, fresh training, archive build from notebook, loader entrypoint check passing on archived main.py, fixture no-crash, local smoke illegal=0 and exceptions=0, runtime feature crosscheck passing, archive contents valid, and holdout quality not materially worse than v0-06d10.

rejection_evidence: Any loader check selecting a helper callable, fixture exception, illegal action, exception, archive missing required files, notebook failure, unexplained holdout regression, stale v0-06d10 artifact reuse, or policy-scope change beyond entrypoint hardening.

go_no_go: go
