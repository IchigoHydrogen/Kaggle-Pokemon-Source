# Pokemon Local Loop Record

## Version Goal

version: `pokemon-20260624-v0-06d12`
baseline: `pokemon-20260624-v0-06d11`
canonical_notebook: `pokemon-20260624-v0-06d12.ipynb`
runtime_mode: guarded_torch_policy
promotion_type_target: runtime_promote

goal:
  Part A — 3つの実装ミスを一括修正:
    (1) BCE loss → per-decision cross-entropy (F.cross_entropy over n options within each decision)
    (2) card_id%32 one-hot hash → 35-dim card metadata (all_card_data(): cardType, evo stage,
        ex/mega/tera/aceSpec flags, HP, 19 key-card identity bits); feature dim stays 96
    (3) ATTACK/END/RETREAT hard veto → confidence-gated override (ABILITY keeps hard veto)
  Part B — Guarded runtime probe with NameError(sys) fix + loader entrypoint validation.

hypothesis:
  CE + card metadata で model_top1 が 0.5090 以上を維持し、
  hybrid_top1 が 0.44 以上になる（PLAY/ATTACH/EVOLVE/RETREAT の confidence-gate override で
  rule の 0.188 から大幅改善）。

change_scope:
  Loss function (BCE→CE), feature encoder ([61-95] card metadata 35-dim),
  safe_override_types (ATTACK/END excluded, RETREAT included),
  runtime sys import fix, SKIP_TRAIN logic.
  Model architecture and MLP dims unchanged.

fresh_training: false (run3: SKIP_TRAIN=True, loaded from run2 model file)

## Pre-Implementation Self-Review

go_no_go: go

scope_guardrails:
  - Feature dim unchanged (96)
  - MLP architecture unchanged
  - Episode split unchanged
  - Guard threshold by grid search on valid split only (no holdout peeking)
  - ABILITY remains hard-vetoed

known_discrepancy:
  - ATTACK/END model accuracy < rule (ATTACK: 0.398 < 0.504, END: 0.063 < 0.154),
    so these types excluded from safe_override_types.

## Implementation & Execution

実装:
- `_v06d12_main_cell.py`: Part A — CE loss, 35-dim card metadata at [61-95],
  V06D12_SAFE_MAIN_OVERRIDE_TYPES = PLAY,ATTACH,EVOLVE,RETREAT (ATTACK/END excluded).
  SKIP_TRAIN block: loads existing .pt if found, skips 30 ep training (~25 min saved).
- `_v06d12_runtime_cell.py`: Part B — frozenset({7,8,9,12}) (PLAY,ATTACH,EVOLVE,RETREAT).
  NameError fix: `sys.argv[1]` → `_v06d12_sys.argv[1]` in crosscheck script f-string.
- `_build_v06d12_nb.py`: builds 27-cell notebook from v06d11.

Run history:
  Run 1: FileNotFoundError (archive not found) — failed before Part B.
  Run 2: All fixes applied, ran to completion. ATTACK/END included in types (old value),
          safe_types=['PLAY','ATTACH','EVOLVE','ATTACK','END','RETREAT'], hybrid=0.3102, prob=0.65.
  Run 3: SKIP_TRAIN fired (loaded run2 model), safe_types corrected ['PLAY','ATTACH','EVOLVE','RETREAT'],
          hybrid=0.3102, prob=0.65.

Bugs found and fixed:
  (a) NameError `sys` in crosscheck f-string: `sys.argv[1]` → `_v06d12_sys.argv[1]`. Fixed before run 3.
  (b) safe_override_types env default included ATTACK,END: fixed `frozenset({7,8,9,12})`. Fixed before run 3.
  (c) SKIP_TRAIN if-block had empty body (SyntaxError): restructured as `if _SKIP_TRAIN:...else:(train)`.

## Final Full Results

モデル品質（holdout, prob_threshold=0.65, margin_threshold=0.0）:
- rule top-1: 0.1877
- model top-1: **0.5095** (baseline v06d11: 0.5090, diff=+0.0005)
- hybrid top-1: **0.3102** (v06d11 baseline: 0.4507, REGRESSION)
- override_rate: 0.195
- selected threshold: prob=0.65, margin=0.0
- ATTACK delta (holdout): -0.0128 (passes -0.03 gate)
- quality_ok: **False** (hybrid 0.31 < 0.44 gate)

Threshold grid (valid split):
- prob=0.35: valid_hybrid=0.4535, danger_hybrid_delta=-0.0598, **danger_ok=False** → selection_ok=False
- prob=0.65: valid_hybrid=0.3077, danger_hybrid_delta=-0.0061, danger_ok=True → selection_ok=True, **selected**
- Root cause: danger gate threshold = -0.01 blocks prob=0.35 even though danger_ok=-0.06 may be acceptable

Hybrid regression root cause analysis:
  Even with ATTACK/END excluded from safe_override_types, at prob=0.35 the model often
  recommends PLAY over ATTACK when they both appear in a decision (model thinks PLAY > ATTACK).
  This overrides correct rule-ATTACK decisions to incorrect PLAY, hurting ATTACK bucket accuracy.
  On holdout: prob=0.35 gives ATTACK_delta=-0.0921; self-review promotion gate is -0.03.
  No threshold satisfies BOTH hybrid_top1 ≥ 0.44 AND ATTACK_delta ≥ -0.03 simultaneously.

  This is a guard logic design issue:
  - Current: gate `model_top1_type IN safe_types` (prevents overriding TO ATTACK/END)
  - Needed: also gate on `rule_type NOT IN [ATTACK,END]` (prevents overriding FROM correct ATTACK)

Guarded Runtime Probe (Part B):
- torch_load_ok: True
- runtime feature crosscheck: passed=True, max_err=0.0 (NameError fix confirmed working)
- archive_files: 11
- safe_override_types runtime: [7, 8, 9, 12] = PLAY,ATTACH,EVOLVE,RETREAT ✓
- hard_veto_types: [10] = ABILITY ✓
- 20 games: illegal_actions=0, exceptions=0, smoke_gate_ok=True
- action_changes: 153/671 (22.8% of decisions)
- latency: p50=0.25ms, p95=0.54ms, p99=0.81ms, max=5.3ms
- loader entrypoint: _kaggle_submission_entrypoint, argcount=2, ok=True
- model_params: 345,473

SKIP_TRAIN:
- Run 3: SKIP_TRAIN=True (epochs_run=0, best_epoch=0, best_valid_top1=0.0)
- Model loaded from: pokemon-20260624-v0-06d12-main_option_scorer.pt (written at run2 06:57)

Hard gates:
- notebook full execution: ✓ (exit code 0)
- smoke_gate_ok: ✓ (illegal=0, except=0)
- loader_entrypoint_ok: ✓
- runtime_feature_crosscheck_passed: ✓
- danger_gate_ok: ✓ (attack_delta=-0.013 ≥ -0.03)
- quality_ok: **✗** (hybrid 0.31 < 0.44)
- all_gates_ok: **False**

## Promotion Decision

decision: **needs_followup**

reason: モデル訓練（CE + card metadata）は正しく機能し model_top1=0.5095 を達成（baseline維持）。
         NameError(sys) バグ・safe_override_types(ATTACK/END除外) の修正は正しく適用された。
         しかし hybrid quality gate が失敗（0.31 < 0.44）。
         原因: danger gate (-0.01) が prob=0.35 (holdout hybrid=0.454) を排除し、
         唯一 selection_ok=True な prob=0.65 (hybrid=0.31) が選択される。
         prob=0.35 では model が ATTACK chosen 決定を PLAY に override し
         ATTACK bucket accuracy が -9.2% 悪化（cross-type contamination）。
         self-review gate の -0.03 も満たさない。
         guarded runtime probe は全ゲート通過（smoke ok, loader ok, crosscheck ok）。

promotion_type: needs_followup

known_findings:
  1. CE + card meta features: model は正しく訓練される。この実装は次版でも保持する。
  2. ATTACK cross-type contamination: モデルが PLAY を選好する場面で ATTACK chosen 決定にも
     影響が及ぶ。現在の danger_mask (chosen_option_type_name) が厳格すぎる。
  3. guard logic fix needed: rule_option_type NOT IN [ATTACK,END] を override 条件に追加することで
     cross-type contamination を排除できる（v06d13 で実装予定）。
  4. SKIP_TRAIN が正常動作：run3 で既存 .pt からロードし ~25 min の再訓練を回避。
  5. danger gate threshold (-0.01) と promotion gate (-0.03) に差異がある。
     selection threshold は promotion threshold に合わせるか、rule-type ベースに変更すべき。

next_candidates:
  - v0-06d13: Fix guard logic — add rule_type NOT IN [ATTACK,END] as override precondition.
              Keep CE loss + card metadata features. Use prob=0.35 for PLAY/ATTACH/EVOLVE/RETREAT.
  - 代替案: type-specific threshold（PLAY/ATTACH/EVOLVE に prob=0.35、RETREAT に prob=0.55）
