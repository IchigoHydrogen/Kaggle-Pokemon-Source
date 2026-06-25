# Pokemon Local Development Loop Harness

See **Baseline Handling** for current runtime and research baseline metric values.

This document defines how Codex should run local improvement loops for the Pokemon TCG AI Battle notebook/codebase. The purpose is to improve the agent using replay/log datasets and local evaluation, without submitting to Kaggle during the loop.

## Core Rule

Do not start an improvement loop until a concrete version goal and validation plan are written down.

Each loop must change one main hypothesis only. If multiple ideas are found, queue them as future candidates instead of mixing them into the current version.

Each version's canonical runnable artifact is a versioned notebook, not a standalone `.py` script. A `.py` file may be used as a companion artifact for diffing, debugging, or command-line execution, but the version must be reproducible by executing the notebook end-to-end.

Each version should train or fit its own model, thresholds, predictions, and policy artifacts from raw or declared datasets. Do not silently reuse prior-version model weights, predictions, threshold outputs, or validation decisions.

Replay/log datasets are the primary evidence for policy learning and promotion. The primary promotion signal is **winner_margin** (defined below), which is measurable from replay data alone without local self-play.

Local games against random, first-valid, or other weak agents are smoke and durability checks only. Their win rates must not be used as a promotion criterion.

Before editing code, Codex must self-review the next experiment plan and implementation plan. This review should tighten the scope, check whether the validation can actually prove the hypothesis, distinguish runtime feasibility from policy-strength evidence, and identify likely failure modes. Implementation starts only after this review is written down.

## Primary Metric: Winner Margin

The primary training signal is **winner_margin**, computed on the holdout split from replay/log data:

```
winner_top1   = fraction of holdout decisions made by the winning player
                where the model's top-1 prediction matches the player's chosen action
loser_top1    = same fraction for decisions made by the losing player
winner_margin = winner_top1 − loser_top1
```

A positive winner_margin means the model is more likely to agree with winners than losers. A larger winner_margin is a proxy for competitive policy value, because the model selectively imitates the strategy of players who actually won.

Rules for winner_margin:

- Always compute winner_margin on the holdout split; never tune against it.
- Report winner_top1 and loser_top1 separately in addition to the margin.
- Compare the model's winner_margin against the rule agent's winner_margin on the same holdout slice as the baseline.
- A version must show winner_margin > baseline_winner_margin to qualify for learning_promote or runtime_promote.
- If winner_margin is zero or negative, the model is not learning winner-selective behavior and should not be promoted.
- Overall holdout_model_top1 (aggregate imitation accuracy) is a sanity check and regression guard, not the promotion criterion.

## Development Tracks

Each version must declare a development track before implementation:

```text
runtime_track
exploration_track
```

Definitions:

- `runtime_track`: A version intended to become, or directly test, a submission/runtime candidate. Runtime safety, archive validity, loader behavior, action-change measurement, and regression gates are mandatory.
- `exploration_track`: A version intended to produce reusable research evidence, identify promising learning signals, or choose the next hypothesis. It does not need to be runtime-ready and should not be treated as a submission candidate unless a later runtime-track version adopts it.

Exploration-track versions should be used to test learning signals that are closer to the target objective than local imitation alone, while clearly recording their limitations and avoiding runtime claims until validated.

Track rules:

- Runtime-track versions must satisfy all applicable hard gates before `runtime_promote`.
- Exploration-track versions may use offline-only, diagnostic-only, or shadow-only analyses when the goal explains why they are useful.
- Exploration-track versions must still be reproducible, use fixed splits, avoid holdout tuning, write prefixed artifacts, and record a promotion decision.
- Exploration-track evidence must be clearly separated from runtime adoption evidence. Do not call an exploration result runtime-ready unless the runtime gates have actually been tested.
- High-risk behavior should not be ignored merely because it is hard to deploy. It may be studied through offline diagnostics, family-specific probes, shadow runtime, or other non-action-changing methods before any runtime adoption.
- A version may compare against both a runtime baseline and a research baseline, but it must state which one is used for which purpose.

## Directory And Naming Rules

### Single-Machine Format

For a source notebook named:

```text
pokemon-YYYYMMDD-vX-YYzz-<machine>.ipynb
```

all generated artifacts must be written under:

```text
/kaggle/working/pokemon-YYYYMMDD-vX-YYzz-<machine>/
```

Every artifact file generated inside that output directory should use the same prefix:

```text
pokemon-YYYYMMDD-vX-YYzz-<machine>-<artifact-name>.<ext>
```

Submission archives in `/kaggle/working` must also be prefixed:

```text
pokemon-YYYYMMDD-vX-YYzz-<machine>-submission.tar.gz
pokemon-YYYYMMDD-vX-YYzz-<machine>-submission-rule-only.tar.gz
pokemon-YYYYMMDD-vX-YYzz-<machine>-submission-table-or-numpy-policy.tar.gz
pokemon-YYYYMMDD-vX-YYzz-<machine>-submission-shadow-torch-policy.tar.gz
pokemon-YYYYMMDD-vX-YYzz-<machine>-submission-guarded-torch-policy.tar.gz
pokemon-YYYYMMDD-vX-YYzz-<machine>-submission-unknown0-policy-table.tar.gz
```

Avoid writing bare `submission.tar.gz` in `/kaggle/working`, because multiple local versions will accumulate there.

The notebook is the canonical version artifact. If a companion `.py` script is generated, it must use the same prefix and must not replace the notebook as the promotion artifact.

Before promotion, the versioned notebook must be executed end-to-end and must regenerate the expected artifacts under the same prefix.

### Multi-Machine Naming Convention

This project is developed across two machines:

| Nickname | Host | GPU |
|---|---|---|
| `ei` | ei@DESKTOP-73HS8PJ | RTX 5070 |
| `remote-pc` | fukuharaken@DESKTOP-T45CBIC | RTX 4090 |

The `<machine>` suffix in the prefix is the machine nickname. Examples:

```text
pokemon-20260625-v0-07d5-ei.ipynb          → RUN_PREFIX = pokemon-20260625-v0-07d5-ei
pokemon-20260625-v0-07d5-remote-pc.ipynb   → RUN_PREFIX = pokemon-20260625-v0-07d5-remote-pc
```

Rules:

- The per-machine iteration counter (e.g., `07d5`) increments **independently per machine**. No cross-machine counter coordination is needed.
- The major version (`vX`) is **shared**. Bumping the major version requires coordination between machines.
- Two experiments with the same date and iteration number but different machine suffixes are separate experiments and do not conflict.
- **Always include the machine suffix** even when only one machine is active, for future clarity.

### Git Coordination Rules

Source control is at `https://github.com/IchigoHydrogen/Kaggle-Pokemon-Source.git`. The working directory `/kaggle/working/` (host: `~/wkdir/2422005/kaggle/working/`) is the git root.

- **Pull from `main` before starting a new version.**
- Push after a version's promotion decision is recorded. Push: notebook + `.md` records + harness updates + companion `.py` files.
- Do not push notebooks whose execution failed or whose promotion decision is missing.
- Tracked files: `*.ipynb`, `*.md`, `*.py` (excluding `tmp_*`), `*-promotion-decision.json`.
- Everything else (archives, weights, data, temp files) is gitignored via `.gitignore`.

## Notebook-First Implementation Rule

Implement each loop in the versioned notebook. The notebook should contain, or directly generate, the training, evaluation, runtime injection, archive packaging, and validation gates needed to reproduce the version.

Do not make a standalone `.py` script the main implementation path. Companion `.py` files may be generated for inspection, debugging, diffing, or temporary command-line checks, but the promoted behavior must be reproducible by executing the canonical notebook end-to-end.

Do not manually fix or repack a prior archive as the promoted artifact. If an archive or generated `main.py` needs a fix, put that fix back into the notebook generation path and rerun the notebook so the current `{RUN_PREFIX}-submission*.tar.gz` is regenerated from the current notebook.

The canonical notebook should be reasonably self-contained for the current version. It may use helper functions, cells, and raw datasets already present in the workspace, but the current version's model weights, predictions, thresholds, generated runtime source, reports, and promotion decision should be produced by the current notebook run under the current `RUN_PREFIX`.

## Version Independence

It is acceptable to copy or bump a previous notebook as the starting point for a new version. After the bump, the new notebook must stand on its own as the canonical artifact for the current version.

A current version must not depend on prior-version generated artifacts for its evidence or runtime behavior unless that reuse is explicitly declared in the version goal and self-review. In particular, do not silently read prior-version:

- model weights,
- predictions,
- selected thresholds,
- generated `main.py`,
- submission archives,
- validation decisions,
- promotion metrics,
- run summaries as current-run evidence.

Allowed reuse includes raw or declared datasets, card metadata, sample submission support files such as `cg/`, baseline source code used as a template, deck definitions, and explicit failure fixtures used only for no-crash or regression checks.

Baseline reports and prior promotion decisions may be read for comparison, but they must remain comparison evidence. The current version's promotion decision must be based on artifacts regenerated under the current `RUN_PREFIX`.

If the version fixes packaging, runtime entrypoint, loader compatibility, or another deployment-path issue, the fix still belongs in the notebook. The notebook must regenerate the corrected archive and record the deployment-path validation result.

## RL Episode Data Policy

ゲームシミュレーションは RL 実験における実行時間の主要ボトルネックである。1 イテレーション 200 試合 × 0.28 秒 / 試合 ≈ 1 分弱となり、訓練規模が大きくなるにつれこのコストは増大する。PDCA サイクルを高速に回すため、RL エピソードデータを再利用可能な成果物として扱う。これは Imitation Learning における人間リプレイの decision_rows と同じ位置づけである。

**デフォルト方針：エピソードデータは複数バージョンで使い回す。**

エピソード収集フェーズと訓練フェーズを分離する：

- **Episode Collection（収集フェーズ）** — behavioral policy でゲームをシミュレーションし、各決定の features・chosen action・log_prob・step rewards・GAE returns・advantages を保存する。成果物として `<RUN_PREFIX>-rl_episodes.parquet`（または同等のファイル）を書き出し、behavioral policy のバージョンと対戦相手構成を記録する。
- **RL Training（訓練フェーズ）** — 保存済みエピソードを読み込んで勾配更新のみ実行する。ゲームシミュレーションは行わない。このフェーズは数分で完了する。

ハイパーパラメータのみを変更する実験（λ_rl・学習率・PPO epochs・clip_epsilon・バッチサイズ・報酬係数）ではエピソード再収集は不要である。バージョン目標の `episode_source` フィールドにソースバージョンを宣言し、既存データを使用する。

**必要になったら保守的にならずに再収集する。**

再収集を過度に避けてはならない。以下の条件に該当する場合は再収集すること：

- モデルアーキテクチャまたは特徴量次元が変わった（behavioral policy の log_prob が無効になり重要度補正が機能しない）。
- 対戦相手の構成またはポリシーが大きく変わった。
- winner_margin が十分に改善し、behavioral policy が現在の policy と比べて著しく弱くなった（分布ずれが訓練を歪める）。
- 報酬構造または step_reward 係数が変わり、既存の advantage 推定が不整合になった。
- 訓練が飽和した（PPO の clip 比率が常に境界に達し、勾配更新がほぼゼロになる）。この飽和は失敗ではなく、新しいシミュレーションデータが必要になったサインである。

**宣言要件。**

すべての RL バージョンはバージョン目標に `episode_source` を記録しなければならない：

- 新規収集の場合：`episode_source: fresh (behavioral policy: <当該バージョン>)`
- 再利用の場合：`episode_source: <ソースバージョン> (behavioral policy: <そのバージョン>; reuse justified: <理由>)`

エピソードデータの無宣言での再利用を禁止する。宣言なしの再利用は分布ずれの評価やリグレッションの追跡を不可能にする。

## Runtime Mode

Each version must declare one runtime mode before implementation:

```text
rule_only
table_or_numpy_policy
shadow_torch_policy
guarded_torch_policy
```

Definitions:

- `rule_only`: final `main.py` is rule logic only and must not import `torch`.
- `table_or_numpy_policy`: final `main.py` may use tables, signatures, thresholds, or NumPy-style lightweight inference, but must not import `torch`.
- `shadow_torch_policy`: final `main.py` may import `torch` for measurement only; model output must not change selected actions. **Shadow mode is a transitional crash-check step**, not a promoted final state. Its purpose is to verify that the model loads without crashing and produces acceptable latency. Once shadow gates pass, the next version should target guarded mode.
- `guarded_torch_policy`: final `main.py` may import `torch` and may change selected actions only through explicit guards and fallback paths. **This is the default deployment target.** Every runtime-track version should aim for guarded mode unless an explicit reason to stay in shadow is stated.

A Torch runtime mode must record whether Torch loads, whether the model file is included in the archive, inference latency, eligible/action-change or shadow-disagreement counts, fallback counts, and safety results. Torch inference must be batched across legal options; do not call the model once per option.

Shadow mode may be used as the first step of a two-version deployment sequence: (1) shadow to verify crash safety and latency, (2) guarded to enable actual action changes and measure winner_margin improvement in competition. Do not stay in shadow mode across more than one sequential version unless a specific safety concern requires it.

## Version Goal Template

Before implementing a new version, create or update a short goal record using this shape:

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
fresh_training:
artifact_reuse_policy:
episode_source:
datasets:
target_metric:
baseline_winner_margin:
success_criteria:
rollback_criteria:
expected_artifacts:
```

The `target_metric` field must name the primary signal being optimized (normally `winner_margin`). The `baseline_winner_margin` field must record the rule agent's or prior version's winner_margin on the same holdout slice, so that improvement can be directly verified.

Example (illustrative only — fill in current values from the Baseline Handling section):

```text
version: pokemon-YYYYMMDD-vX-YYzz
baseline: <current research baseline; see Baseline Handling section>
canonical_notebook: pokemon-YYYYMMDD-vX-YYzz.ipynb
development_track: runtime_track | exploration_track
runtime_mode: guarded_torch_policy
promotion_type_target: runtime_promote | learning_promote | exploration_promote
goal: <one sentence describing what this version accomplishes>
hypothesis: <one-sentence causal claim that the version tests>
change_scope: <what specifically changes; list anything that does NOT change>
fresh_training: true | false (reuse <source>; declared reuse)
artifact_reuse_policy: <what is reused and from where; "none" if all regenerated>
datasets: <which replay/log datasets>
target_metric: winner_margin (holdout_winner_top1 − holdout_loser_top1)
baseline_winner_margin: <current research baseline value from Baseline Handling section>
success_criteria:
- Notebook executes end-to-end.
- holdout winner_margin > baseline_winner_margin.
- <any mode-specific or experiment-specific gate>
rollback_criteria:
- winner_margin <= baseline_winner_margin.
- <any hard safety condition>
expected_artifacts:
- executed notebook
- main_model_report.json (with winner_top1, loser_top1, winner_margin)
- promotion-decision.json
- prefixed submission archive (if runtime_track)
```

## Pre-Implementation Self-Review

Before code edits for a new version, write a short self-review using this shape:

```text
experiment_plan:
implementation_plan:
why_this_is_the_next_best_step:
what_would_make_this_result_untrustworthy:
expected_failure_modes:
scope_guardrails:
validation_plan:
promotion_evidence_required:
rejection_evidence:
go_no_go:
```

The review must answer:

- Is the hypothesis narrow enough for one version?
- Does the implementation plan test that hypothesis directly?
- Is the canonical notebook target named?
- Is the development track named, and is the validation plan appropriate for that track?
- Is the runtime mode named?
- Is fresh training or artifact reuse explicit?
- Is there a simpler diagnostic that should be run before changing behavior?
- What is the expected winner_margin improvement, and how is it computed?
- Which metrics could improve while the actual agent gets worse?
- Which buckets are most likely to regress?
- Can validation distinguish runtime/action-path success from policy-strength evidence?
- If this is an exploration-track version, what evidence would make the result useful without making it runtime-ready?
- What exact evidence is needed to promote the version?
- What exact evidence should stop the loop or reject the version?

Rules:

- If the self-review finds that the plan mixes multiple hypotheses, split it before editing code.
- If the validation cannot distinguish a real improvement from noise, improve the validation first.
- If the validation only proves runtime feasibility, do not call it a strength improvement.
- If the validation only produces exploration evidence, do not call it runtime-ready.
- If the change mainly optimizes a visible metric without a winner_margin check, do not implement it.
- If the likely failure mode is severe, add a targeted stress check before promotion.
- Do not proceed from planning to implementation without an explicit `go`.
- Do not proceed if the plan silently relies on prior model weights, predictions, or thresholds.

## Fixed Data Split Policy

Use deterministic splits by `episode_id`, replay filename hash, or stable game identifier hash. Do not use an unseeded random split.

Recommended split:

```text
train: 70%
valid: 15%
holdout: 15%
```

Within the holdout split, compute winner_top1 and loser_top1 separately using the `won` column in the decision rows. Both sub-slices must have at least 200 samples to be used as promotion evidence; otherwise mark as low-confidence.

Add a separate stress slice when possible:

```text
stress:
- UNKNOWN_0 decisions
- contexts or action families touched by the version
- dangerous contexts such as ATTACK or END when relevant
- low-frequency matchups
- long games
- losing games
- high-rank/top200 games
- cases with previous fallback or disagreement
```

Rules:

- `train` is for fitting heuristics, policies, models, thresholds, and candidate decks.
- `valid` is for choosing among alternatives inside the current version.
- `holdout` is for promotion decisions only.
- `stress` is for safety and regression checks.
- Do not tune directly against `holdout`.
- If a bucket has too few samples, mark it as low-confidence instead of using it as promotion evidence.

## Required Gates

A version cannot be promoted unless all applicable hard gates pass:

- Canonical notebook executes to completion.
- Executed notebook is saved.
- `main.py` compiles.
- Importing `main.py` does not require `deck.csv` at import time.
- Final selected deck has exactly 60 integer card IDs.
- Local smoke evaluation has zero illegal actions.
- Local smoke evaluation has zero exceptions.
- Submission archive contains `main.py`, `deck.csv`, `cg/api.py`, and `cg/game.py`.
- Submission archive does not contain `__pycache__`, `.pyc`, or `.pyo`.
- Artifacts are written under `/kaggle/working/<RUN_PREFIX>/`.
- Main output artifacts use `<RUN_PREFIX>-` filename prefix.
- Submission archives use `<RUN_PREFIX>-submission*.tar.gz`.
- Promotion decision record is written.
- Fresh training and artifact reuse status are recorded.
- Runtime mode is recorded.
- **winner_top1, loser_top1, winner_margin are computed and recorded on the holdout split.**
- **For learning_promote and runtime_promote: holdout winner_margin > baseline_winner_margin.**

Archive and loader gates:

- Validate the actual archived `main.py`, not only an in-memory source string or imported module.
- `main.py` in the final archive must compile after extraction.
- For Kaggle raw Python submissions, validate the same raw-loader path used by `kaggle_environments` when practical, such as `kaggle_environments.agent.get_last_callable`.
- Record the callable selected by the raw-loader validation.
- If the selected callable is expected to be a wrapper or entrypoint, verify its name, arity, and no-crash behavior.
- If bundled runtime model files are required, verify model loading from the extracted archive under the raw-loader path.
- If a hosted validation failure fixture is available, use it as an archive-level no-crash regression check. The fixture is for runtime validation only and must not be used for training or threshold tuning.

Mode-specific gates:

- `rule_only`: final submission source must not import `torch`.
- `table_or_numpy_policy`: final submission source must not import `torch`; missing table/model artifacts must fall back to rule logic.
- `shadow_torch_policy`: Torch must load, model file must be archived, output must not change actions, latency and shadow disagreement must be recorded. Shadow mode is crash-check only and does not satisfy the action-change requirement for runtime_promote.
- `guarded_torch_policy`: Torch must load, model file must be archived, inference must be batched, fallback must exist, latency and action-change stats must be recorded, and smoke checks must have zero illegal actions and zero exceptions. **action_changes must be > 0** to confirm the model is actually modifying decisions.

A run with zero action changes may be useful as a shadow diagnostic, but it cannot be promoted as an action-changing guarded policy.

## Evaluation Metrics

Do not judge a version by a single aggregate score.

The primary promotion evidence, in priority order:

1. **winner_margin** = holdout_winner_top1 − holdout_loser_top1. This is the main promotion criterion.
2. **holdout_winner_top1**: How accurately the model matches winning-side decisions.
3. **holdout_loser_top1**: How accurately the model matches losing-side decisions (should be lower than winner_top1).
4. **holdout_model_top1**: Aggregate imitation accuracy. Used as a regression guard; must not degrade severely.

Track in addition:

- Illegal action count.
- Exception count.
- Fallback count and fallback rate.
- Runtime per game.
- Per-action latency p50, p95, p99, and max when relevant.
- action_changes count and rate (for guarded mode; must be > 0 to confirm guarded operation).
- UNKNOWN_0 policy hit rate, miss rate, and selected action distribution when relevant.
- Metrics by rank bucket: top10, top50, top200, other when available.
- Metrics by matchup bucket when available.
- Metrics by select context and action family touched by the version.
- Metrics for dangerous buckets touched by the version.
- Local smoke performance against random or simple agents (crash and legality check only).

Dangerous buckets include contexts or action families where a bad choice can immediately lose the game, such as ATTACK, END, targeting, attachment, search, discard, recovery, and deckout-risk decisions. Only require detailed dangerous-bucket reporting when the version touches or may affect those buckets.

Exploration-track versions may also use research evidence that is not itself runtime adoption evidence, such as:

- Objective-proxy metrics justified by the version goal.
- Outcome-aware or disagreement-based diagnostics.
- Family-specific or context-specific recovery metrics.
- Slice consistency across rank, matchup, strength bucket, game phase, or other declared partitions.
- Valid-to-holdout consistency of a candidate signal.
- Shadow-only or offline-only disagreement measurements.
- Clear identification of the next hypothesis and what remains unproven.

Rules for exploration evidence:

- Treat exploration evidence as prioritization or research evidence unless a runtime-track version validates runtime adoption.
- Do not present a proxy metric as direct proof of runtime strength.
- Do not tune against holdout, even for exploration.
- Mark low-sample slices as low-confidence.
- Record what the result supports, what it does not support, and which follow-up hypothesis it enables.

## Promotion Decision

At the end of each version, record a decision:

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
hard_gates:
runtime_baseline:
research_baseline:
baseline_comparison:
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

Promotion type definitions:

- `runtime_promote`: the generated submission policy is safe enough to become the next runtime baseline. Requires guarded mode with action_changes > 0.
- `learning_promote`: the training/evaluation method, model, or diagnostic is useful and should be carried forward. Winner_margin has improved over the baseline, but the version is not yet deployed in guarded mode.
- `exploration_promote`: the version is not runtime-ready but produced reproducible research evidence that should guide a future hypothesis or research baseline.
- `needs_followup`: the run is technically valid but more diagnosis is required before promotion.
- `reject`: the version should not be carried forward.

Runtime promotion requires:

- The version is a runtime-track version, or it explicitly reruns a runtime-track validation before promotion.
- All applicable hard gates pass.
- Canonical notebook executes end-to-end.
- **Runtime mode is guarded_torch_policy with action_changes > 0.**
- **holdout winner_margin > baseline_winner_margin.**
- holdout_model_top1 does not severely regress.
- No severe bucket-level regression.
- Dangerous buckets touched by the version do not regress materially.
- Any improvement is explainable from artifacts.
- Runtime behavior is measured.
- The code diff remains reversible.
- Artifact reuse is absent or explicitly justified.
- Local weak-agent win rate is not the only positive evidence.

Learning promotion requires:

- Canonical notebook executes end-to-end.
- Training and evaluation artifacts are reproducible.
- Fresh training or declared artifact reuse is clear.
- **holdout winner_margin > baseline_winner_margin** (or the version provides clear evidence that the winner_margin signal is being improved, with an explicit explanation if direct comparison is not possible due to distribution shift).
- Holdout and stress or bucket metrics are reported.
- Known runtime blockers are recorded.
- The result is not incorrectly labeled as runtime-ready.

Exploration promotion requires:

- Canonical notebook executes end-to-end.
- The declared exploration evidence is regenerated under the current `RUN_PREFIX`.
- Fresh training or declared artifact reuse is clear.
- The result is checked on holdout or another declared non-training evaluation slice.
- The evidence is not only a train/valid artifact.
- winner_top1, loser_top1, and winner_margin are reported where computable.
- Slice, rank, family, context, matchup, or other declared consistency checks are reported when relevant.
- Low-sample or noisy evidence is marked as low-confidence.
- The record states what the result supports and what it does not prove.
- A concrete next hypothesis is recorded.
- The result is not labeled as a runtime baseline or submission candidate.

Reject when:

- Safety gates fail.
- Notebook execution fails.
- Archive generation fails.
- `main.py` fails to compile.
- Local smoke eval has illegal actions.
- Local smoke eval has exceptions.
- **holdout winner_margin <= baseline_winner_margin** (for learning_promote or runtime_promote targets).
- holdout_model_top1 severely regresses.
- Dangerous buckets regress.
- Improvement only appears on train/valid.
- Metrics are too noisy to justify promotion.
- The version changes too many independent things.
- Prior weights, predictions, or thresholds are reused without declaration.
- guarded mode has action_changes = 0 (model is not actually changing decisions).

Use `needs_followup` when the run is technically valid but more diagnosis is required before promotion.

## Loop Procedure

For each version:

0. **Meta-cognitive framing (mandatory before anything else).** Before writing the version goal or touching code, articulate in plain language:
   - What the current state is (runtime baseline, research baseline, known gaps).
   - What the new harness formally requires for the target promotion type (winner_margin gate, guarded mode gate, etc.).
   - What specific gap this experiment is closing, and what gaps will remain after it.
   - What this experiment is actually testing versus what it is merely changing.
   - Which causal assumptions underlie the hypothesis, and where those assumptions could be wrong.
   - What your own likely bias is in proposing this experiment (incremental safety bias, metric-gaming bias, etc.), and how the design resists it.
   - What a negative result would look like, and whether you could distinguish it from a positive one.
   This framing must be written down before proceeding to step 1. It is not a checklist to rush through; it is the reasoning that determines whether the experiment is worth running at all.

1. Read the relevant runtime baseline and, if applicable, research baseline summaries and artifacts. Record baseline_winner_margin from the research baseline.
2. Write the version goal using the template above. Include `target_metric` and `baseline_winner_margin`.
3. Write the pre-implementation self-review.
4. If the self-review says `go`, copy or bump the notebook name and `RUN_PREFIX`.
5. Make the smallest implementation change that tests the hypothesis.
6. Run a smoke check.
7. Execute the canonical notebook end-to-end.
8. Verify that artifacts were regenerated under `/kaggle/working/<RUN_PREFIX>/`.
9. Verify that generated artifacts use the `<RUN_PREFIX>-` prefix.
10. Compute winner_top1, loser_top1, winner_margin on the holdout split. Compare against baseline_winner_margin.
11. Compare holdout_model_top1 against baseline as a regression guard.
12. Run local smoke/durability checks for legality, exceptions, packaging, and runtime.
13. Record the promotion decision including winner_margin_summary.
14. Queue future hypotheses separately.

For exploration-track versions:

- Runtime action changes are optional and should usually be disabled unless the goal is a shadow or runtime probe.
- Record the exploration evidence, its limitations, and the next hypothesis.
- Record whether the result should become a research baseline.
- Do not replace the runtime baseline unless a later runtime-track version validates the behavior.

Do not continue automatically into the next version unless explicitly instructed.

If a run fails, Codex may fix the failure only when the fix is necessary to validate the current hypothesis and does not broaden the scope. If the failure reveals that the hypothesis, dataset, or validation design is flawed, record the failure and stop instead of silently changing the experiment.

## Long-Running Execution Monitoring

Notebook execution can take several minutes, especially during replay parsing, feature extraction, training, local games, or archive validation. During long-running execution, avoid excessive polling and repeated broad directory scans.

Recommended monitoring behavior:

- Check promptly after starting execution to confirm the process began.
- During expected long cells, check at coarse intervals rather than continuously. A practical default is every 2-5 minutes; use longer intervals for known slow feature extraction or training cells when CPU or output files indicate progress.
- Prefer small, targeted checks: process still running, latest relevant artifacts, or the final report file.
- Avoid repeatedly dumping large JSON files, full directory listings, notebook outputs, or archive contents while execution is still in progress.
- Increase monitoring detail only when there is an error, timeout, no CPU activity, no file changes for an unexpectedly long period, or another sign that the run is stuck.
- After execution finishes, perform the full verification once: notebook exit status, required artifacts, reports, archive contents, loader gates, smoke gates, and promotion decision.

The goal is to preserve enough visibility to catch failures without spending excessive time or tokens on low-value progress checks.

For AI agent operation, use `ScheduleWakeup(300)` immediately after launching a background notebook execution. At each wakeup:

1. Check whether the nbconvert process is still running (`ps aux | grep nbconvert | grep -v grep`).
2. Check the log tail for the latest printed output line (one targeted `tail -5` is enough).
3. Check the newest artifact timestamp in the output directory (`ls -lt <output_dir>/ | head -3`).
4. If still running: schedule the next `ScheduleWakeup(300)` and stop.
5. If finished: proceed to full verification (artifacts, reports, gates, promotion decision).

This prevents context loss when a notebook run outlasts a single agent turn. Do not wait for the run to finish before scheduling the first wakeup.

## Baseline Handling

Version names are **not** recorded in this section. To identify which version established a baseline, check the corresponding `*-promotion-decision.json` or version goal file. Recording only metric values here means that updating a baseline requires changing numbers in one place, with no version-name strings scattered through the harness.

### Runtime Baseline Metrics

```text
holdout_model_top1: 0.509
runtime_mode:       guarded_torch_policy
```

### Research Baseline Metrics

```text
winner_margin (holdout, stored-feature eval):  0.057
holdout_winner_top1:                           0.478
holdout_loser_top1:                            0.422
il_baseline_winner_margin:                    -0.004
feature_dim: 97  (dim 96 = won-conditioning; 1.0 at inference, actual won-status in stored features)
evaluation_note: "stored-feature" eval preserves dim 96 from the training feature matrix
  (1.0 for winner decisions, 0.0 for loser decisions). An earlier reference value of 0.050
  appeared in prior harness versions; it was computed by a different method and is no
  longer the operative baseline.
source_version_note: v07d4 offline PPO on v07d3 episodes (8 epochs). Smoke gate not
  confirmed due to SKIP_PIPELINE+Cell[25] incompatibility (import_agent_from_source
  missing). Model quality metric is valid; submission packaging pending fix in v07d5.
```

When updating the research baseline after a `learning_promote`, replace only the metric block above. Do not add version names.

The research baseline may differ from the runtime baseline. A research baseline is a reproducible version or report used as the starting point for exploration, diagnostics, or learning-method development. It is not automatically a submission candidate.

Before replacing the baseline, keep:

- the executed canonical notebook,
- any companion scripts,
- `/kaggle/working/<RUN_PREFIX>/` artifact directory,
- final run summary,
- training report,
- validation or holdout report,
- runtime report,
- generated `main.py`,
- generated `deck.csv`,
- final submission archive,
- promotion decision record.

Only a `runtime_promote` version becomes the next runtime baseline.

A version with `learning_promote` may become the new research baseline for future experiments. Update the research baseline in this harness when a learning_promote version shows winner_margin improvement.

A version with `exploration_promote` may become a research baseline or may define the next exploration hypothesis. It must not replace the runtime baseline unless a later runtime-track version adopts and validates the behavior through runtime gates.

When a version uses both baselines, record:

- which runtime baseline is used for safety or deployment comparison,
- which research baseline is used for exploration comparison,
- which artifacts are comparison evidence only,
- whether the result changes neither baseline, only the research baseline, or the runtime baseline.

## Two-Agent Experiment Type Policy

Each machine independently tracks whether its current experiment is **conservative** or **aggressive**.

### Definitions

- **conservative**: Builds on the current known-best approach with incremental changes. Examples: hyperparameter tuning, offline PPO with saved episodes, adding features to the existing model, packaging fixes, IL data extensions, threshold recalibration, or minor architecture tweaks.
- **aggressive**: Introduces a substantially different approach that cannot be validated by small adjustments to the current baseline. Examples: new model architecture, new learning algorithm (e.g., switching from PPO to REINFORCE or value-based), entirely new feature family, new reward formulation, large structural refactor, or a policy strategy from a different design paradigm.

### The 2-Conservative Rule

**After 2 consecutive conservative experiments on a machine, the next experiment on that machine must be aggressive.**

This rule is mandatory, not advisory. It prevents the loop from stagnating on incremental polish when a more exploratory approach is needed for long-term gains.

- The count resets to 0 whenever an aggressive experiment is run (regardless of promotion outcome).
- The count is **per-machine**. One machine running conservative while the other runs aggressive is acceptable.
- "Consecutive" means back-to-back on the same machine, regardless of what the other machine is doing.

### Required Fields in Version Goal and Promotion Decision

Add these fields to every version goal and promotion decision:

```text
experiment_type: conservative | aggressive
consecutive_conservative_count: N   # value BEFORE this experiment (0 if this is aggressive or first)
```

If `consecutive_conservative_count` reaches 2 and this experiment is marked `conservative`, that is a harness violation. Stop, reconsider the plan, and design an aggressive experiment instead.

### Scope of "Aggressive"

Aggressive does not mean reckless. The same self-review, fixed split, winner_margin gate, and promotion criteria apply. "Aggressive" refers to the hypothesis design space, not to skipping validation. An aggressive experiment that fails is still a valid exploration_promote result if it produces reproducible evidence for a future hypothesis.

## Codex Operating Constraints

Codex should follow these constraints during improvement loops:

- **Before any other step, write the meta-cognitive framing (Loop Procedure step 0). Do not skip this even if the next experiment feels obvious.**
- State the version goal before editing code.
- State the canonical notebook before editing code.
- State the development track before editing code.
- Self-review the experiment and implementation plan before editing code.
- Do not proceed if the self-review does not produce an explicit `go`.
- Do not silently broaden scope.
- **Use winner_margin (holdout_winner_top1 − holdout_loser_top1) as the primary promotion criterion.**
- **Do not promote a version whose winner_margin does not exceed baseline_winner_margin.**
- Do not optimize against only one visible metric.
- Do not promote based only on local random/simple-agent win rate.
- Do not treat runtime feasibility as policy-strength proof.
- Do not treat exploration evidence as runtime adoption evidence.
- Do not replace the runtime baseline with an exploration result.
- Do not reuse prior weights, predictions, or thresholds unless explicitly declared.
- **RL episode data may be reused across versions; declare `episode_source` in the version goal.** Regenerate when architecture, feature dimension, opponent policy, or reward structure changes materially. Do not avoid regeneration out of excessive caution — if the data is stale, collect fresh episodes.
- Do not delete prior version artifacts unless explicitly asked.
- Do not submit to Kaggle as part of this loop.
- **Shadow mode is a transitional crash-check step, not a final promoted state. Move to guarded mode as soon as shadow gates pass.**
- **Guarded mode (action_changes > 0) is the default deployment target for runtime-track versions.**
- Prefer replay/log datasets for training and promotion evaluation.
- Use local weak agents only for smoke, legality, packaging, and runtime checks.
- Keep every run reproducible through prefix, seed, split manifest, and artifact records.
- Keep runtime baseline and research baseline roles explicit when they differ.
- Keep `.ipynb` as the canonical runnable artifact for each version.
- Use `.py` only as a companion artifact unless explicitly instructed otherwise.
- Put fixes to generated runtime source, packaging, archive contents, or Kaggle loader compatibility back into the canonical notebook and regenerate the archive from that notebook.
- Keep each version reasonably independent: do not require prior-version generated artifacts to reproduce the current version unless that dependency is explicitly declared and justified.
- Monitor long-running notebook execution at coarse, useful intervals rather than repeatedly polling with broad commands.
- If a run fails, fix the failure only when it is necessary to validate the current hypothesis; otherwise record the failure and stop.
- Update the research baseline section of this harness whenever a learning_promote or runtime_promote version shows winner_margin improvement. Update only the metric values in **Baseline Handling**; do not add version names there.
- **Notebook inter-cell interface contract.** Cell [23] (or the equivalent training/RL cell) must define the following notebook-global variables before it exits, or Cell [25] (runtime probe and promotion decision) will fail:
  - `MAIN_HYBRID_REPORT` — dict with at minimum: `quality_ok` (bool), `all_gates_ok` (bool), `winner_margin` (float), `model` (dict with `model_path`, `best_epoch`, `epochs_run`, `best_valid_top1`, `param_count`), `holdout_summary` (dict with `model_top1`), `holdout_hybrid_summary` (dict with `hybrid_top1`).
  - `MAIN_LEARNING_REPORT` — dict with training log and final metric values.
  - `_card_table` — dict mapping `int(card_id)` → card object, required for Cell [25]'s feature cross-check.
  Any variable renamed in Cell [23] (e.g. `_v07d2_card_table` instead of `_card_table`) must be aliased before the cell exits.
