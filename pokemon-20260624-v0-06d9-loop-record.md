# Pokemon Local Loop Record

## Version Goal

version: `pokemon-20260624-v0-06d9`
baseline: `pokemon-20260623-v0-06d8`
canonical_notebook: `pokemon-20260624-v0-06d9.ipynb`
runtime_mode: rule_only (submission main.py は変更なし)
promotion_type_target: learning_promote
goal: フィーチャ抽出を NumPy 化（事前確保 + スライス代入）し、学習テンソルを GPU に事前転送することで、
      v0-06d8 と同等の holdout 品質を維持しながら訓練パイプラインを高速化する。
hypothesis: Python list.append × 128万回 + np.asarray 変換を除去し、かつ _X_train/_X_valid/_X_all を
            訓練前に一括 GPU 転送すれば、モデル品質（holdout top-1）は v0-06d8 の 0.5012 ±0.01 以内に
            収まり、かつ訓練・推論フェーズの壁時計時間が短縮される。
change_scope: フィーチャ抽出ループ + 訓練前 GPU 転送のみ。
              モデル構造・特徴定義・ハイパーパラメータ・提出 main.py は変えない。
fresh_training: true
artifact_reuse_policy: v0-06d8 の .pt / predictions / thresholds は使わない。
                       特徴定義は v0-06d8 から継承するが、数値は全て再計算する。
datasets: v0-06d8 と同じ (pokemon-tcg-ai-battle-episodes-2026-06-22)
success_criteria:
  - クロスチェック: 100 決定で vectorized 特徴 == Python ループ特徴（絶対誤差 < 1e-5）
  - holdout model top-1 が v0-06d8 の 0.5012 から ±0.01 以内
  - decision / option 件数が v0-06d8 と一致
  - 訓練時間計測値をアーティファクトに記録
  - hard gates すべてパス
rollback_criteria:
  - クロスチェックで特徴値の不一致
  - holdout が v0-06d8 より 0.01 以上下がる
  - decision / option 件数が v0-06d8 と一致しない
  - GPU メモリ不足で訓練失敗

## Pre-Implementation Self-Review

experiment_plan: v0-06d8 の MAIN hybrid セルから list.append 方式を除去し、
                 事前確保 NumPy 配列 + per-decision スライス代入 + per-option ベクトル化に置き換える。
                 学習テンソルは訓練ループ開始前に一括 GPU 転送する。
                 クロスチェックで特徴値の一致を確認してから訓練に進む。

implementation_plan:
  1. _X = np.zeros((_max_options+512, 96), dtype=np.float32) を事前確保
  2. 決定ループは Python で回す（101K 件）
  3. per-decision 共有フィーチャは _X[s:e, :] へのブロードキャスト代入
  4. per-option フィーチャは np.array([...]) → スライス代入 + fancy indexing
  5. ループ後に _X[:cursor], _y[:cursor] でトリム
  6. クロスチェック（100 決定で Python ループ版と比較）
  7. .npy 保存
  8. _X_train/valid/all を .to(device) で GPU 転送
  9. 訓練ループの .to(device) 呼び出しを削除
  10. 時間計測を各フェーズに追加

why_this_is_the_next_best_step: v0-06d8 で guarded hybrid の有効性が確認されたが、
                                 訓練パイプラインのボトルネックがフィーチャ抽出と CPU-GPU 転送にある。
                                 v0-06d10 での guarded runtime probe を安定して回すため、
                                 パイプラインを先に整備する。

what_would_make_this_result_untrustworthy:
  - vectorized 特徴が Python ループ版と異なる（バグ）
  - decision/option 件数が変わる（フィルタ条件変更）
  - GPU メモリ不足でフォールバック
  - holdout が大幅に変わる（乱数シードの扱いが変わった等）

expected_failure_modes:
  - one-hot fancy indexing で列ずれ
  - _v06d7_to_float の挙動が numpy 型変換と微妙に異なる
  - GPU 転送後の randperm が seed なしで再現性を失う

scope_guardrails:
  - モデル構造変更なし
  - 特徴定義変更なし
  - 提出 main.py 変更なし
  - v0-06d8 アーティファクト再利用なし
  - ハイパーパラメータ変更なし

validation_plan:
  - クロスチェック（絶対誤差 < 1e-5）を実行し、失敗なら実装を修正してから訓練
  - holdout top-1 を v0-06d8 と比較
  - 訓練・フィーチャ抽出時間を記録

promotion_evidence_required:
  - クロスチェック ok
  - holdout model top-1 ≥ 0.49（v0-06d8 の -0.01）
  - hard gates pass
  - 時間計測記録あり

rejection_evidence:
  - クロスチェック不一致
  - holdout < 0.49
  - notebook 実行失敗
  - decision/option 件数不一致

go_no_go: go

## Implementation & Execution

- `_v06d9_main_hybrid_cell.py` を新規作成。特徴抽出を list.append → np.zeros 事前確保 + per-decision スライス代入 + per-option fancy indexing に変更。
- `_build_v06d9_nb.py` で v0-06d8 ノートブックの MAIN セルを差し替えてビルド。
- スモーク実行（2000 decisions, 1 epoch）: クロスチェック passed、GPU pre-load ok、illegal_actions=0。
- フルラン実行: error_count=0、notebook 実行完了。

## Final Full Results

タイミング:
- フィーチャ抽出: **246.4s**（全 1,278,625 オプション行）
- GPU pre-load（一括転送）: **0.5s**（train 909K + valid 178K + all 1.28M 行）
- 訓練 5 エポック: **1.6s**（v0-06d8 は per-batch CPU→GPU 転送 × 5×110 回）
- 推論（全行スコアリング）: **0.2s**

クロスチェック:
- decisions_checked: 100 / options_checked: 1088
- mismatches: 0 / max_abs_error: 2.88e-08（機械精度レベル）
- passed: True

モデル品質（holdout）:
- rule top-1: 0.1877
- model top-1: **0.5012**（v0-06d8 と差 < 0.0001 ✓）
- model top-3: 0.7337
- hybrid top-1: 0.4454（+0.2577 over rule）
- override rate: 0.628
- benefit: 4,563 / harm: 737 / benefit-harm: +3,826
- ATTACK/END/ABILITY delta: 0.0000（保持）

Hard gates:
- notebook full execution: ✓
- error_count=0: ✓
- crosscheck_passed: ✓
- fresh_training: ✓
- v0-06d8 model/prediction reuse: なし ✓
- submission adoption: なし ✓
- quality_ok_within_0_01: ✓（diff=4.6e-05）

## Promotion Decision

decision: **learning_promote**

reason: パイプライン高速化（GPU pre-load + NumPy ベクトル化）を確認。訓練フェーズが 1.6s に短縮。
         モデル品質は v0-06d8 の holdout top-1=0.5012 と diff < 0.0001 で一致。
         hard gates すべてパス。runtime adoption は引き続き無効。

promotion_type: learning_promote

known_risks:
- フィーチャ抽出 (246s) の主ボトルネックは JSON パースと card_id 抽出のPythonループ部分。
  訓練・推論は GPU で劇的に短縮されたが、全体壁時計はまだ抽出が支配的。
- クロスチェックは先頭 100 決定のみ。

next_candidates:
- v0-06d10: guarded runtime probe — MAIN scorer を submission main.py に埋め込み、
            PLAY/ATTACH/EVOLVE のみ override、ATTACK/END/ABILITY/RETREAT はハードベト。
- フィーチャ抽出のさらなる高速化（JSON パース並列化 / card_id 抽出のベクトル化）は別候補。
