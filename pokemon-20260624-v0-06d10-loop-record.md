# Pokemon Local Loop Record

## Version Goal

version: `pokemon-20260624-v0-06d10`
baseline: `pokemon-20260624-v0-06d9`
canonical_notebook: `pokemon-20260624-v0-06d10.ipynb`
runtime_mode: guarded_torch_policy
promotion_type_target: runtime_promote

goal:
  Part A — Early stopping: valid top-1 で patience=5、max 100 epoch で訓練し最良チェックポイントを保存。
  Part B — Guarded runtime probe: 学習済み MAIN scorer を提出用 main.py に組み込み
            PLAY/ATTACH/EVOLVE のみ override、ATTACK/END/ABILITY/RETREAT をhard veto として
            guarded_torch_policy アーカイブを生成・ローカル検証する。

hypothesis:
  (A) 5 epoch 固定より早期終了チェックポイントの holdout top-1 が同等以上になる
  (B) guarded_torch_policy アーカイブでローカル smoke eval が
      illegal_actions=0 / exceptions=0 / torch_load=ok を満たす

change_scope:
  訓練ループ（early stopping + best checkpoint）と
  提出 main.py への torch guard 注入のみ。
  モデル構造・特徴定義・ハイパーパラメータ・guard 閾値は v0-06d9 を引き継ぐ。

fresh_training: true
runtime_mode_change: rule_only → guarded_torch_policy

## Pre-Implementation Self-Review

go_no_go: go

scope_guardrails:
  - モデル構造・特徴定義変更なし
  - v0-06d9 アーティファクト再利用なし（再訓練）
  - guard 閾値は v0-06d9 の valid 選択値を引き継ぐ
  - RETREAT を runtime veto に追加（offline eval との差分を記録）
  - モデルデバイスは CPU 固定

known_discrepancy:
  offline eval (v0-06d9) の rule_veto_types には RETREAT が含まれていないが、
  runtime probe では RETREAT も hard veto とする。
  これにより実際の override_rate は offline eval の 0.628 より若干低くなる。

## Implementation & Execution

実装:
- `_v06d10_main_hybrid_cell.py`: Part A — early stopping on valid top-1 (patience=5, max 100 ep)。
  GPU pre-loading + 全特徴抽出ロジックは v0-06d9 を継承。
  best_state_dict を CPU 保存し、ループ後に restore。
- `_v06d10_guarded_runtime_cell.py`: Part B — trained model を rule-only main.py に注入。
  SAFE_OVERRIDE_TYPES={7,8,9}、HARD_VETO_TYPES={10,12,13,14}。
  runtime feature crosscheck (synthetic obs, subprocess)、guarded torch archive ビルド、
  local smoke eval (latency tracking)。
- `_build_v06d10_nb.py`: v0-06d9 ノートブックから v0-06d10 を生成。
  cell 23 を Part A で差し替え、cell 24-25 に Part B markdown+code を挿入（計27セル）。

スモーク実行: V06D10_MAIN_MAX_DECISIONS=2000, 3エポック → Part A ok, Part B ok (illegal=0)。
  スモーク時: epochs_run=3 best_epoch=2 best_valid_top1=0.40 holdout=0.43 quality_ok=False（件数制限、想定通り）
  Part B スモーク: torch_load=True cc_passed=True illegal=0 except=0 latency_p99=0.58ms gate_ok=True

フルラン実行完了:
- Part A: status=ok error_count=0
- Part B: status=ok
- アーカイブ再ビルド: Zone.Identifier ファイル除外（`'Zone.Identifier' in n` filter）

## Final Full Results

タイミング:
- フィーチャ抽出: **278.1s**（全 1,278,625 オプション行 / 101,253 decisions）
- GPU pre-load: **0.3s**
- 訓練（early stopping）: **4.9s**（14 epochs / best epoch 8）
- 推論: **0.19s**

クロスチェック:
- 訓練側 (vectorized): max_abs_error=2.88e-08, passed=True
- ランタイム側 (synthetic obs): max_err=0.0, passed=True

Early stopping:
- epochs_run: 14 / max 100
- best_epoch: 8
- best_valid_top1: **0.5043**（v0-06d9 最終 valid は 5ep固定で未計測）

モデル品質（holdout）:
- rule top-1: 0.1877
- model top-1: **0.5027**（v0-06d9: 0.5012、diff=+0.0015）
- hybrid top-1: **0.4466**（+0.2589 over rule, v0-06d9: 0.4454 diff=+0.0012）
- override_rate: 0.631
- benefit: 4,574 / harm: 730 / net: +3,844
- selected threshold: prob=0.35 margin=0.0
- ATTACK delta: 0.0 / END delta: 0.0 / ABILITY delta: 0.0 (harm=0, benefit=0)
- RETREAT delta: NaN（chosen RETREAT bucket では harm=2/benefit=0; runtime は RETREAT hard veto）

Guarded Runtime Probe:
- torch_load_ok: True
- runtime feature crosscheck: passed=True max_err=0.0
- 20 games: illegal_actions=0, exceptions=0, smoke_gate_ok=True
- action_changes: 306/557 (54.9% of MAIN decisions)
- latency: p50=0.20ms, p95=0.33ms, p99=0.58ms, max=4.10ms

Hard gates:
- notebook full execution: ✓ (exit code 0)
- error_count=0: ✓
- crosscheck_passed (train + runtime): ✓
- fresh_training: ✓
- smoke_gate_ok: ✓ (illegal=0, except=0, torch_load=True)
- quality_ok: ✓ (holdout model_top1=0.5027 ≥ 0.5012-0.01=0.4912)
- all_gates_ok: ✓

提出アーカイブ:
- `pokemon-20260624-v0-06d10-submission.tar.gz` (guarded_torch_policy, 2,350,344 bytes)
  - main.py (rule agent + v06d10 guard injection)
  - deck.csv
  - cg/{__init__.py, api.py, cg.dll, game.py, libcg.so, sim.py, utils.py}
  - models/main_option_scorer.pt
  - Zone.Identifier ファイルなし ✓
- `pokemon-20260624-v0-06d10-submission-rule-only.tar.gz` (rule-only, 従来比較用)
- `pokemon-20260624-v0-06d10-submission-unknown0-policy-table.tar.gz` (UNKNOWN0 policy)

## Promotion Decision

decision: **runtime_promote**

reason: v0-06d9 の learning_promote から全ゲートを通過し、guarded_torch_policy での runtime 採用が可能。
         早期終了チェックポイント（best epoch 8, valid_top1=0.5043）により holdout model top-1 が 0.5012→0.5027 に微改善。
         hybrid top-1 も 0.4454→0.4466 に向上。
         ATTACK/END/ABILITY delta=0.0（保持）。
         RETREAT は hard veto で override なし。
         Guarded runtime probe: torch load ok, feature crosscheck ok, 20局 illegal=0/except=0, latency p99=0.58ms。
         提出アーカイブに models/main_option_scorer.pt を内包。

promotion_type: runtime_promote

known_risks:
- Kaggle 提出環境での torch import が実際に成功するかは未確認（ユーザー確認済みの情報に依拠）。
  torch load 失敗時は `_V06D10_MODEL_LOADED=False` となり rule-only フォールバックで動作。
- 訓練フィーチャ抽出は 278s と依然ボトルネック（JSON パース Python ループ）。
- offline eval の rule_veto_types（ATTACK/END/ABILITY）と runtime hard_veto_types（ATTACK/END/ABILITY/RETREAT）に差分あり。
  RETREAT の offline eval 評価では harm=2/benefit=0 だが件数が少なく（n=368）影響軽微。

next_candidates:
- v0-06d11: Kaggle コンペ提出 + 実際の対戦成績の確認
- フィーチャ抽出高速化（JSON パース並列化）
- モデルアーキテクチャ改善（プレイヤー状態の embedding 化等）
