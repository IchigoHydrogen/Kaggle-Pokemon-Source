# Pokemon Local Development Loop Harness

Rules and criteria for the Pokemon TCG AI Battle improvement loop.
Operational procedures → `loop-skills.md`. Experiment queue → `experiment-backlog.md`.

See **Baseline Handling** for current metric values.

---

## Core Rule

Do not start an improvement loop until a concrete version goal and validation plan are written down.

Each loop must change one main hypothesis only. Queue additional ideas in `experiment-backlog.md` instead of mixing them into the current version.

Each version's canonical runnable artifact is a versioned notebook. A companion `.py` may be used for diffing or debugging but must not replace the notebook as the promotion artifact.

Each version must train or fit its own model, thresholds, predictions, and policy artifacts from raw or declared datasets. Do not silently reuse prior-version model weights, predictions, threshold outputs, or validation decisions.

Replay/log datasets are the primary evidence for policy learning and promotion. The primary promotion signal is **winner_margin**, measurable from replay data alone without local self-play.

Local games against random, first-valid, or other weak agents are smoke and durability checks only. Their win rates must not be used as a promotion criterion.

---

## Primary Metric: Winner Margin

```
winner_top1   = fraction of holdout decisions made by the winning player
                where the model's top-1 prediction matches the player's chosen action
loser_top1    = same fraction for decisions made by the losing player
winner_margin = winner_top1 − loser_top1
```

A positive winner_margin means the model is more likely to agree with winners than losers.

Rules:

- Always compute winner_margin on the holdout split; never tune against it.
- Report winner_top1 and loser_top1 separately in addition to the margin.
- Compare against baseline_winner_margin from the **Baseline Handling** section.
- A version must show winner_margin > baseline_winner_margin to qualify for `learning_promote` or `runtime_promote`.
- If winner_margin is zero or negative, the model is not learning winner-selective behavior and must not be promoted under those types.
- Overall holdout_model_top1 is a sanity check and regression guard, not the promotion criterion.

---

## Development Tracks

Each version must declare one development track:

- **`runtime_track`**: Intended to become a submission/runtime candidate. Runtime safety, archive validity, loader behavior, action-change measurement, and all regression gates are mandatory.
- **`exploration_track`**: Intended to produce reusable research evidence or identify the next hypothesis. Does not need to be runtime-ready. Must not be labeled as a submission candidate unless a later runtime-track version adopts and validates it.

Track rules:

- Runtime-track versions must satisfy all applicable hard gates before `runtime_promote`.
- Exploration-track versions may use offline-only, diagnostic-only, or shadow-only analyses when the goal explains why they are useful.
- Exploration-track versions must still: be reproducible, use fixed splits, avoid holdout tuning, write prefixed artifacts, record a promotion decision.
- Do not call an exploration result runtime-ready unless runtime gates have actually been tested.
- A version may compare against both a runtime baseline and a research baseline, but must state which is used for which purpose.

---

## Directory And Naming Rules

For a source notebook named `pokemon-YYYYMMDD-vX-YYzz-<machine>.ipynb`:

- All generated artifacts must be written under `/kaggle/working/pokemon-YYYYMMDD-vX-YYzz-<machine>/`
- Every artifact file must use the prefix `pokemon-YYYYMMDD-vX-YYzz-<machine>-<artifact-name>.<ext>`
- Submission archives must use `pokemon-YYYYMMDD-vX-YYzz-<machine>-submission*.tar.gz`

Do not write bare `submission.tar.gz` in `/kaggle/working`.

The notebook is the canonical version artifact. Companion `.py` scripts must use the same prefix and must not replace the notebook as the promotion artifact.

Before promotion, the versioned notebook must execute end-to-end and regenerate expected artifacts under the same prefix.

Multi-machine naming convention and git workflow are in `loop-skills.md`.

---

## Notebook-First Implementation Rule

Implement each loop in the versioned notebook. The notebook should contain or directly generate: training, evaluation, runtime injection, archive packaging, and validation gates needed to reproduce the version.

Do not make a standalone `.py` script the main implementation path.

Do not manually fix or repack a prior archive as the promoted artifact. If a fix is needed, put it back into the notebook generation path and rerun.

The canonical notebook should be reasonably self-contained. Helper functions and raw datasets from the workspace are allowed, but the current version's model weights, predictions, thresholds, generated runtime source, reports, and promotion decision must be produced by the current notebook run under the current `RUN_PREFIX`.

---

## Version Independence

A current version must not depend on prior-version generated artifacts unless explicitly declared in the version goal. In particular, do not silently read prior-version:

- model weights, predictions, selected thresholds, generated `main.py`,
- submission archives, validation decisions, promotion metrics,
- run summaries as current-run evidence.

Allowed without declaration: raw or declared datasets, card metadata, `cg/` support files, deck definitions, baseline source code used as a template, explicit failure fixtures for no-crash or regression checks.

Declared reuse must be stated in `artifact_reuse_policy` in the version goal. Baseline reports and prior promotion decisions may be read for comparison but must remain comparison evidence only.

---

## RL Episode Data Policy

Game simulation is the primary runtime bottleneck. Separate collection from training:

- **Collection phase**: run game simulation with the behavioral policy; save features, chosen actions, log_prob, step rewards, GAE returns, advantages. Write as `<RUN_PREFIX>-rl_episodes.pt`. Record behavioral policy version and opponent mix.
- **Training phase**: load saved episodes; run gradient updates only; no game simulation.

**Default: reuse episodes across versions.** Recollect when:

- Model architecture or feature dimension changed (stored log_probs are invalid; importance correction fails).
- Opponent composition or policy changed materially.
- winner_margin improved enough that the behavioral policy is significantly weaker than the current policy.
- Reward structure or step_reward coefficients changed, making stored advantage estimates inconsistent.
- Training saturated (PPO clip ratio consistently at boundary; gradient updates near zero).

Do not avoid recollection out of excessive caution. Stale data that distorts training is worse than the cost of recollection.

**Declaration required** in every RL version goal:

- Fresh: `episode_source: fresh (behavioral policy: <this version>)`
- Reuse: `episode_source: <source version> (behavioral policy: <that version>; reuse justified: <reason>)`

---

## Runtime Mode

Each version must declare one runtime mode:

- **`rule_only`**: final `main.py` must not import `torch`.
- **`table_or_numpy_policy`**: may use tables/NumPy; must not import `torch`; missing artifacts must fall back to rule logic.
- **`shadow_torch_policy`**: `torch` may be imported for measurement only; model output must not change actions. **Transitional crash-check step only.** Move to guarded mode as soon as shadow gates pass. Do not stay in shadow across more than one sequential version without a specific safety reason.
- **`guarded_torch_policy`**: `torch` may change actions through explicit guards and fallback paths. **Default deployment target.** `action_changes` must be > 0 to confirm the model is modifying decisions.

Torch runtime modes must record: whether Torch loads, model file included in archive, inference latency p50/p95/p99, action-change or shadow-disagreement counts, fallback counts, safety results. Torch inference must be batched across legal options; do not call the model once per option.

A run with zero action changes may be useful as a shadow diagnostic but cannot be promoted as an action-changing guarded policy.

---

## Required Gates

A version cannot be promoted unless all applicable gates pass.

**Universal gates:**

- Canonical notebook executes to completion.
- Executed notebook is saved.
- `main.py` compiles.
- Importing `main.py` does not require `deck.csv` at import time.
- Final selected deck has exactly 60 integer card IDs.
- Artifacts written under `/kaggle/working/<RUN_PREFIX>/`.
- Artifact filenames use `<RUN_PREFIX>-` prefix.
- Submission archives use `<RUN_PREFIX>-submission*.tar.gz`.
- Promotion decision record is written.
- Fresh training and artifact reuse status are recorded.
- Runtime mode is recorded.
- **winner_top1, loser_top1, winner_margin computed on holdout split.**

**Smoke and archive gates:**

- Local smoke evaluation: zero illegal actions.
- Local smoke evaluation: zero exceptions.
- Submission archive contains `main.py`, `deck.csv`, `cg/api.py`, `cg/game.py`.
- Submission archive does not contain `__pycache__`, `.pyc`, `.pyo`.
- Validate the actual archived `main.py` after extraction, not only the in-memory string.
- Record the callable selected by the Kaggle loader validation (expected: `_kaggle_submission_entrypoint`).

**Promotion-type gates:**

- `learning_promote` / `runtime_promote`: **holdout winner_margin > baseline_winner_margin.**
- `runtime_promote`: runtime mode must be `guarded_torch_policy` with `action_changes > 0`.
- `rule_only`: final submission source must not import `torch`.
- `shadow_torch_policy`: Torch loads, model archived, output does not change actions, latency recorded.
- `guarded_torch_policy`: Torch loads, model archived, inference batched, fallback exists, latency and action-change stats recorded, smoke: zero illegal actions and zero exceptions.

---

## Evaluation Metrics

Primary promotion evidence, in priority order:

1. **winner_margin** — main promotion criterion.
2. **holdout_winner_top1** — accuracy on winning-side decisions.
3. **holdout_loser_top1** — accuracy on losing-side decisions (should be lower than winner_top1).
4. **holdout_model_top1** — aggregate imitation accuracy; regression guard only.

Also track: illegal action count, exception count, fallback count and rate, runtime per game, latency p50/p95/p99/max, action_changes count and rate (guarded mode), UNKNOWN_0 policy hit/miss rate, metrics by rank bucket (top10/top50/top200/other), by matchup, by action family touched by the version.

**Dangerous buckets** — require detailed reporting when the version touches them: ATTACK, END, targeting, attachment, search, discard, recovery, deckout-risk decisions.

Exploration-track versions may also use: objective-proxy metrics, outcome-aware or disagreement-based diagnostics, family-specific or context-specific recovery metrics, slice consistency checks, shadow-only or offline-only disagreement measurements. Record what the result supports, what it does not support, and which follow-up hypothesis it enables.

---

## Fixed Data Split Policy

Use deterministic splits by episode_id, replay filename hash, or stable game identifier hash. Do not use an unseeded random split.

```
train:   70%
valid:   15%
holdout: 15%
```

Within the holdout split, compute winner_top1 and loser_top1 separately using the `won` column. Both sub-slices need ≥ 200 samples to count as promotion evidence; otherwise mark low-confidence.

Stress slices (add when possible): UNKNOWN_0 decisions, action families touched by the version, dangerous contexts (ATTACK, END), low-frequency matchups, long games, losing games, high-rank/top200 games, prior-fallback or prior-disagreement cases.

Split roles:
- `train`: fitting heuristics, policies, models, thresholds, candidate decks.
- `valid`: choosing among alternatives within the current version.
- `holdout`: promotion decisions only. Do not tune against it.
- `stress`: safety and regression checks.

---

## Version Goal Template

```text
version:
machine:
experiment_type: conservative | aggressive
consecutive_conservative_count:
baseline:
canonical_notebook:
development_track:
runtime_mode:
promotion_type_target:
goal:
hypothesis:
change_scope:
fresh_training: true | false
artifact_reuse_policy:
episode_source:
datasets:
target_metric: winner_margin (holdout_winner_top1 − holdout_loser_top1)
baseline_winner_margin:
success_criteria:
rollback_criteria:
expected_artifacts:
```

---

## Pre-Implementation Self-Review Template

```text
meta_cognitive_framing:
hypothesis_check:
change_by_change_review:
scope_check:
failure_modes:
validation_plan:
promotion_evidence_required:
rejection_evidence:
go_no_go:
```

The review must confirm:

- Hypothesis is narrow enough for one version.
- Implementation plan tests that hypothesis directly.
- Canonical notebook, development track, and runtime mode are named.
- Fresh training or artifact reuse is explicit.
- Expected winner_margin improvement is stated and how it is computed.
- Buckets most likely to regress are identified.
- Validation can distinguish runtime success from policy-strength evidence.
- Exact evidence needed to promote is stated.
- Exact evidence that should stop the loop is stated.

Do not proceed to implementation without an explicit `go`.

---

## Promotion Decision Template

```text
decision: promote | reject | needs_followup
promotion_type: runtime_promote | learning_promote | exploration_promote | reject | needs_followup
reason:
machine:
experiment_type: conservative | aggressive
consecutive_conservative_count:
development_track:
runtime_mode:
fresh_training:
artifact_reuse:
hard_gates: pass | fail (list any failures)
runtime_baseline:
research_baseline:
winner_margin_summary:
  holdout_winner_top1:
  holdout_loser_top1:
  holdout_winner_margin:
  baseline_winner_margin:
  improvement:
holdout_summary:
stress_or_bucket_summary:
runtime_summary:
exploration_summary:
known_risks:
next_candidates:
```

**Promotion type definitions:**

- `runtime_promote`: policy is safe enough to become the next runtime baseline. Requires `guarded_torch_policy` with `action_changes > 0` and `holdout winner_margin > baseline_winner_margin`.
- `learning_promote`: training/evaluation method is useful; winner_margin improved over baseline; not yet deployed in guarded mode.
- `exploration_promote`: not runtime-ready but produced reproducible research evidence guiding a future hypothesis.
- `needs_followup`: technically valid but more diagnosis required before promotion.
- `reject`: should not be carried forward.

Reject when: safety gates fail, notebook execution fails, archive generation fails, `main.py` fails to compile, smoke eval has illegal actions or exceptions, `holdout winner_margin ≤ baseline_winner_margin` (for learning/runtime target), `holdout_model_top1` severely regresses, guarded mode has `action_changes = 0`, improvement only appears on train/valid, metrics are too noisy to justify promotion.

Use `needs_followup` when technically valid but more diagnosis is required.

---

## Baseline Handling

Update only the metric values here when a version is promoted. Do not add version names. To identify which version set a baseline, check the corresponding `*-promotion-decision.json`.

### Runtime Baseline

```text
holdout_model_top1: 0.509
runtime_mode:       guarded_torch_policy
```

### Research Baseline

```text
winner_margin (holdout, stored-feature eval):  0.057
holdout_winner_top1:                           0.478
holdout_loser_top1:                            0.422
il_baseline_winner_margin:                    -0.004
feature_dim: 97
evaluation_note: "stored-feature" eval preserves dim 96 from the training feature matrix
  (1.0 for winner decisions, 0.0 for loser decisions).
source_note: offline PPO on saved episodes. Smoke gate not confirmed due to
  SKIP_PIPELINE + Cell[25] incompatibility (import_agent_from_source missing from Cell[3]).
  Model quality metric is valid; submission packaging pending fix.
```

When updating after `learning_promote` or `runtime_promote`, replace only the metric block above.

The research baseline may differ from the runtime baseline. A research baseline guides exploration and is not automatically a submission candidate. Only `runtime_promote` updates the runtime baseline.
