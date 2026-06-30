## Version Goal

version:          pokemon-20260625-v0-05d8
machine:          ei
experiment_type:  aggressive
consecutive_conservative_count: 0
baseline:         pokemon-20260625-v0-05d7
canonical_notebook: pokemon-20260625-v0-05d8.ipynb
development_track: exploration_track
runtime_mode:     table_or_numpy_policy
promotion_type_target: learning_promote

goal: >
  MLP reranker を MAIN コンテキストで初めて submission に有効化する (案B)。
  同時に全コンテキストの per-context winner_margin を診断する (案A)。
  主仮説: MAIN コンテキストで MLP reranker を使うと winner_margin が改善する。
  副産物: 全コンテキストの winner_margin 地図を得る。

hypothesis: >
  MAIN は最頻出かつ最大訓練サンプルのコンテキスト。
  MLP reranker が winner の行動をより多く予測しているなら
  winner_margin (全コンテキスト holdout) は v0-05d7 の UNKNOWN_0 限定 0.081 より
  広い面積で正の値を示す。

change_scope:
  CHANGED:
    Cell[1]: USE_MLP_RERANKER_FOR_SUBMISSION = True
    Cell[1]: MLP_SUBMISSION_CONTEXTS = ['MAIN']
    Cell[1]: RUN_PREFIX -> pokemon-20260625-v0-05d8
    Cell[1]: EXPERIMENT_NAME -> v0_05d8_mlp_main_activate
    Cell[21]: per-context winner_margin 診断ブロック追加
      (MLP reranker / BC の各コンテキスト別に
       holdout 上で winner_top1 / loser_top1 / winner_margin を計算)
  UNCHANGED:
    エピソードソース (episodes-2026-06-24), top200 (06-24),
    UNKNOWN_0 policy table logic, MLP_ALPHA=0.10, MLP_EPOCHS,
    architecture, feature engineering

fresh_training: true
artifact_reuse_policy: なし
episode_source: fresh (pokemon-tcg-ai-battle-episodes-2026-06-24)
datasets:
  - /kaggle/input/competitions/pokemon-tcg-ai-battle-episodes-2026-06-24
  - /kaggle/input/competitions/top200-20260624-ranking/top200-20260624-ranking.csv

target_metric: winner_margin (holdout, MAIN コンテキスト MLP 予測)
baseline_winner_margin: N/A (MAIN コンテキストでの初計測)
secondary_metric: per-context winner_margin map (全コンテキスト)

success_criteria:
  - ノートブック end-to-end 実行完了
  - MAIN コンテキストの winner_margin > 0
  - per-context winner_margin が全コンテキスト分出力される
  - holdout_correct_rate_on_hits >= 0.55 (UNKNOWN_0 policy 回帰なし)
  - MLP が実際に submission main.py に組み込まれている

rollback_criteria:
  - MAIN コンテキストの winner_margin < 0 -> MLP_SUBMISSION_CONTEXTS=[] に戻す
  - holdout_correct_rate_on_hits < 0.50 (policy 品質の大幅劣化)
  - 実行エラー

expected_artifacts:
  - pokemon-20260625-v0-05d8-v05_run_summary.json
  - pokemon-20260625-v0-05d8-per_context_winner_margin.json  (新規)
  - pokemon-20260625-v0-05d8-unknown0_policy_table.json
  - pokemon-20260625-v0-05d8-promotion-decision.json
  - pokemon-20260625-v0-05d8-submission*.tar.gz
