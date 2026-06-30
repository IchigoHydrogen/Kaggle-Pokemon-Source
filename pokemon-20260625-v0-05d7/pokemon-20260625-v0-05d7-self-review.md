## Pre-Implementation Self-Review

### meta_cognitive_framing
v0-05d6 は leakage-free な diagnostic baseline を確立した。
今回はそのまま新データに切り替えるだけで、ロジック変更なし。
バイアスチェック: データを変えるだけなのに promotion を焦る必要はない。
winner_margin の初計測が主目的。

### hypothesis_check
仮説: 06-24 データで policy quality が維持される。
反証可能: holdout_correct_rate_on_hits と winner_margin を数値で確認できる。
仮説は一つ (data freshness effect)。winner_margin 計測は観測追加であり別仮説ではない。

### change_by_change_review

**Cell[1]: EPISODE_ROOT_CANDIDATES 変更**
- [Path('/kaggle/input/competitions/pokemon-tcg-ai-battle-episodes-2026-06-24')] のみに制限。
- find_episode_files() は再帰的に全 .json を取得するため、このパスが存在しなければ
  episode_files=0 になる。パス確認済み (ホスト側 4596 files)。
- 06-22 と 06-24 が混在しないことを確認済み。

**Cell[1]: TOP200_CSV_PATH 変更**
- top200-20260624-ranking.csv に更新。カラム名は同一形式を想定。
  異なる場合は load_top200() のリネーム処理が対応済み。

**Cell[1]: RUN_PREFIX / EXPERIMENT_NAME 変更**
- アーティファクトが v05d6 と混在しないことを保証。

**Cell[21]: winner_margin 追加**
- UNKNOWN0_POLICY_DECISION_EVAL_DF を使用。won 列が存在することを確認済み。
- policy_hit=True かつ split='holdout' のサブセットで計算。
- サンプル数が少ない (<200) 場合は low_confidence フラグを立てる。
- V05_RUN_SUMMARY に 'unknown0_winner_margin' キーで記録。

### scope_check
変更は Cell[1] (4行) と Cell[21] (追加ブロック) のみ。
他セルへの影響なし。

### failure_modes
1. 06-24 の episode ファイルが読めない → find_episode_files() が空を返す → 
   episode_files=0 で RUN_SUMMARY に記録 → 即座に気づける。
2. top200 CSV のカラム変更 → load_top200() でエラー → セルが止まる。
3. UNKNOWN0_POLICY_DECISION_EVAL_DF に won 列なし → KeyError → 
   try/except でスキップしてエラーログに記録。

### validation_plan
- run_summary.json の episode_index_rows で取り込みエポック数を確認。
- unknown0_policy_summary.json の entries と holdout_correct_rate_on_hits を確認。
- unknown0_winner_margin (V05_RUN_SUMMARY 内) を記録。
- submission archives の内容を確認。

### promotion_evidence_required
- exploration_promote の条件: end-to-end 実行完了 + winner_margin 計測成功 + 
  holdout_correct_rate_on_hits >= 0.55

### rejection_evidence
- holdout_correct_rate_on_hits < 0.50 かつ winner_margin 負
- policy_table_entries < 20
- 実行エラーが多数

### go_no_go
**go** — 変更スコープが極めて小さく、失敗した場合も原因が明確。
winner_margin の初計測は v0-05d 系統の昇格基準整合に必須。
