## Version Goal

version:          pokemon-20260625-v0-05d7
machine:          ei
experiment_type:  conservative
consecutive_conservative_count: 1
baseline:         pokemon-20260623-v0-05d6
canonical_notebook: pokemon-20260625-v0-05d7.ipynb
development_track: exploration_track
runtime_mode:     rule_only / table_or_numpy_policy
promotion_type_target: exploration_promote

goal: >
  データ更新実験。エピソードソースを episodes-2026-06-24 に切り替え、
  top200 も top200-20260624-ranking に更新。ロジック変更なし。
  UNKNOWN_0 policy hits に対する winner_margin を v0-05d 系統で初めて計測。

hypothesis: >
  06-24 の新エピソードはより新しいメタを反映しており、
  policy table quality (hit_rate / correct_rate_on_hits) は 0.55 以上を維持。
  winner_margin (UNKNOWN_0 holdout ヒット) は正となる。

change_scope:
  CHANGED:
    Cell[1]: EXPERIMENT_NAME, TOP200_CSV_PATH, EPISODE_ROOT_CANDIDATES, RUN_PREFIX
    Cell[21]: winner_margin 計測ブロック追加 (won 列 x split=holdout x policy_hit=True)
  UNCHANGED:
    policy table logic, MLP reranker, margin_threshold=0.2, min_table_acc=0.75,
    feature signatures, archive packaging

fresh_training: true
artifact_reuse_policy: なし
episode_source: fresh (pokemon-tcg-ai-battle-episodes-2026-06-24)
datasets:
  - /kaggle/input/competitions/pokemon-tcg-ai-battle-episodes-2026-06-24
  - /kaggle/input/competitions/top200-20260624-ranking/top200-20260624-ranking.csv

target_metric: winner_margin (UNKNOWN_0 holdout ヒット: winner_top1 - loser_top1)
baseline_winner_margin: N/A (v0-05d 初計測)

success_criteria:
  - end-to-end 実行完了
  - holdout_correct_rate_on_hits >= 0.55
  - winner_margin が計算される
  - policy_table_entries >= 30
  - 全アーティファクトが v05d7 プレフィックスで出力

rollback_criteria:
  - holdout_correct_rate_on_hits < 0.50 かつ winner_margin 負 → データ統合を検討
  - policy_table_entries < 20
  - 実行エラー

expected_artifacts:
  - pokemon-20260625-v0-05d7-v05_run_summary.json
  - pokemon-20260625-v0-05d7-unknown0_policy_table.json
  - pokemon-20260625-v0-05d7-promotion-decision.json
  - pokemon-20260625-v0-05d7-submission*.tar.gz
  - 実行済みノートブック
