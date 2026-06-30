## Pre-Implementation Self-Review

### meta_cognitive_framing
v0-05d7 で UNKNOWN_0 winner_margin=+0.081 を確認した。
今回は「MLP reranker を MAIN に有効化する」という攻めの一手。
MLP はすでに訓練済みで、変更コストは Cell[1] の 2 フラグのみ。
バイアスチェック: 攻め実験なので失敗時のロールバック基準を明確に持つ。

### hypothesis_check
仮説: MAIN コンテキストで MLP reranker を使うと winner_margin が改善する。
反証可能: holdout の MAIN コンテキスト winner_margin が計測できる。
仮説は一つ (MAIN MLP 有効化)。per-context 診断は計測の追加であり別仮説ではない。

### change_by_change_review

**Cell[1]: USE_MLP_RERANKER_FOR_SUBMISSION = True**
- MLP_ALPHA=0.10 が blending weight。
  rule スコアと MLP スコアを 0.9:0.1 でブレンドして最終選択。
- MLP_SUBMISSION_CONTEXTS=['MAIN'] で MAIN のみに限定。
  他コンテキストは引き続き rule のみ。
- MLP が存在しない場合 (submission_safety 確認未) → 既存の安全ゲートが
  fallback する設計。確認要。

**Cell[1]: MLP_SUBMISSION_CONTEXTS = ['MAIN']**
- MAIN は最頻出コンテキスト。エラーの影響面積が最大だが、
  MLP が最も多くのサンプルで訓練されているコンテキストでもある。

**Cell[21]: per-context winner_margin 診断**
- MLP_VALID_PREDICTIONS_DF / ALAKAZAM_BC_MODELS の holdout 予測を使用。
  (同じ holdout split 上で計算 → チューニングではなく評価)
- コンテキストごとに: n_winner, n_loser, winner_top1, loser_top1, winner_margin を計算。
- サンプル数 < 100 のコンテキストは low_confidence フラグ。
- per_context_winner_margin.json として保存。

### scope_check
Cell[1] と Cell[21] のみ変更。submission ビルドロジック自体は変更なし
(USE_MLP_RERANKER_FOR_SUBMISSION フラグの既存処理に委ねる)。

### failure_modes
1. MLP が MAIN で winner_margin 負 → 即 reject、次は別コンテキスト試行
2. MLP の submission 組み込みが safety gate でブロックされる → 
   mlp_submission_safety.json を確認して対応
3. per-context 診断で MLP_VALID_PREDICTIONS_DF が空 → スキップ処理で継続

### validation_plan
1. per_context_winner_margin.json で MAIN の winner_margin を確認
2. v05_run_summary.json の mlp_reranker_report で MLP が有効かを確認
3. mlp_submission_safety.json で MLP が submission に組まれたかを確認
4. UNKNOWN_0 holdout_correct_rate_on_hits が 0.55 以上を確認 (回帰チェック)

### go_no_go
**go** — MLP はすでに訓練済み、変更は 2 フラグと診断追加のみ。
失敗時の根拠 (per-context winner_margin) が同一 run で得られるため
次の手が即座に決まる。
