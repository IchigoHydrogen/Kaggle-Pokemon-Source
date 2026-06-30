## Version Goal

version:          pokemon-20260625-v0-05d9
machine:          ei
experiment_type:  aggressive
consecutive_conservative_count: 0
baseline:         pokemon-20260625-v0-05d7  (winner_margin=0.081, MC なし)
canonical_notebook: pokemon-20260625-v0-05d9.ipynb
development_track: exploration_track
runtime_mode:     table_or_numpy_policy
promotion_type_target: learning_promote

goal: >
  Monte Carlo リターン重みを MLP 訓練・UNKNOWN_0 policy table fitting に適用し、
  winner_margin への効果を定量的に測定する。
  weight(t) = γ^(n_steps - t)  (γ=0.98)
  終盤ほど重みが大きく、序盤ノイズを相対的に抑制する。
  MLP submission は OFF (v0-05d7 と同等) に戻し、MC weighting の効果のみを測定。

hypothesis: >
  序盤決定（勝敗因果が薄い）と終盤決定（因果が強い）を同重みで学習すると
  シグナルが薄まる。MC return 重み付けで終盤を優先学習すると
  winner_margin (UNKNOWN_0 policy hits, holdout) が v0-05d7 の 0.081 を超える。

change_scope:
  CHANGED:
    Cell[1]:
      - EXPERIMENT_NAME -> v0_05d9_mc_return_weighting
      - RUN_PREFIX -> pokemon-20260625-v0-05d9
      - MC_DISCOUNT_GAMMA = 0.98  (新規)
      - USE_MC_RETURN_WEIGHTS = True  (新規)
      - USE_MLP_RERANKER_FOR_SUBMISSION = False  (d8 から戻す: isolation のため)
      - MLP_SUBMISSION_CONTEXTS = []  (d8 から戻す)
    Cell[7]:
      - DECISION_ROWS_DF に mc_step_weight カラムを追加
        weight = MC_DISCOUNT_GAMMA ^ max(0, n_steps - step)
        (episode_index から n_steps を join)
    Cell[16] (MLP reranker):
      - 訓練ループの loss 計算に mc_step_weight を乗算
        (option row は decision の step_weight を継承)
    Cell[18] (UNKNOWN_0 policy table):
      - policy table fitting で signature ごとの action 選択を
        mc_step_weight による weighted count に変更
        (late-game UNKNOWN_0 決定を優先的にシグナルとする)
    Cell[21]:
      - phase-stratified winner_margin を追加計測
        early (step <= T/3), mid (T/3 < step <= 2T/3), late (step > 2T/3)
      - MC なし baseline (v0-05d7: 0.081) との比較を明記
  UNCHANGED:
    エピソードソース, top200 ソース, MLP アーキテクチャ,
    feature engineering, policy table 閾値 (margin_threshold=0.2, min_table_acc=0.75),
    split 定義

fresh_training: true
artifact_reuse_policy: なし
episode_source: fresh (pokemon-tcg-ai-battle-episodes-2026-06-24)
datasets:
  - /kaggle/input/competitions/pokemon-tcg-ai-battle-episodes-2026-06-24
  - /kaggle/input/competitions/top200-20260624-ranking/top200-20260624-ranking.csv

target_metric: winner_margin (UNKNOWN_0 holdout policy hits)
baseline_winner_margin: 0.081 (v0-05d7, MC なし)
success_criteria:
  - ノートブック end-to-end 実行完了
  - winner_margin > 0.081 (MC なし baseline を超える)
  - phase-stratified 分析で late-game winner_margin > early-game winner_margin
    (MC weighting が期待通りに機能していることの傍証)
  - holdout_correct_rate_on_hits >= 0.55 (回帰なし)
rollback_criteria:
  - winner_margin <= 0.081 (MC weighting 効果なし)
  - holdout_correct_rate_on_hits < 0.50
  - 実行エラー
expected_artifacts:
  - pokemon-20260625-v0-05d9-v05_run_summary.json
  - pokemon-20260625-v0-05d9-mc_phase_winner_margin.json  (新規)
  - pokemon-20260625-v0-05d9-unknown0_policy_table.json
  - pokemon-20260625-v0-05d9-promotion-decision.json
  - pokemon-20260625-v0-05d9-submission*.tar.gz
