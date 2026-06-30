# Pre-Implementation Self-Review: v0-06d18

experiment_plan: >
  Add won as feature dim 96 (0/1 at train, 1.0 always at inference).
  Filter episodes to Top200 vs Top200. Uniform CE unchanged. Measure
  holdout top-1, best_epoch, winner/loser breakdown vs v06d15.

implementation_plan: >
  Base: v06d15 notebook. Cell 0: update EXPERIMENT_NAME, RUN_PREFIX.
  Cell 19: (a) _V06D15_FEAT_DIM 96->97; (b) in extraction loop add
  _X[_s_row:_e_row, 96] = 1.0 if _won else 0.0; (c) reference function
  _v06d15_option_features: add won=True param, set row[96]=float(won);
  (d) cross-check buffer: store bool(_won); (e) episode-level Top200 filter
  before steps loop; (f) feature spec dim 96 doc; (g) rc_bc_report JSON.
  Cell 20: Linear(96,512)->Linear(97,512); both _v06d15_features_batch
  copies: (0,96)->(0,97), (n,96)->(n,97), add X[:,96]=1.0; _synth_ref:
  add won=True.

why_this_is_the_next_best_step: >
  v06d16/v06d17 showed loss weighting creates gradient conflict. Adding won
  as input feature is a cleaner way to condition on outcome: uniform CE,
  all data, no conflict. Top200 filter targets higher-quality decisions.
  Together these test whether outcome conditioning + data quality improve
  over v06d15 (0.5074).

what_would_make_this_result_untrustworthy:
  - Top200 filter leaves < 5000 train decisions (underfitting from data size).
  - won feature is ignored by model (coefficient near zero).
  - best_epoch <= 4 again (underfitting regardless of won feature).
  - Holdout improvement < 0.003 (noise).

expected_failure_modes:
  - Too few Top200 vs Top200 games in dataset (data starvation).
  - won=1.0 at inference creates distribution shift if training distribution
    had many won=0 examples (model never saw won=1 for some states).
  - Cross-check fails due to won feature mismatch between train and runtime.

scope_guardrails:
  - No architecture depth change (same 3-layer MLP).
  - No feature changes in dims [0-95] (identical to v06d15).
  - No deck changes.
  - No runtime action changes (shadow, action_changes must = 0).
  - No holdout tuning.
  - Exactly two changes: won feature + Top200 filter.

validation_plan: >
  Execute v06d18 notebook end-to-end. Verify: 97-dim cross-check passed,
  Top200 filter stats (episodes passed/skipped), train_decisions >= 5000,
  holdout top-1 vs v06d15 (0.5074), best_epoch > 4, winner/loser holdout
  breakdown. Shadow probe: action_changes=0.

promotion_evidence_required: >
  learning_promote: holdout > v06d15 (0.5074) by >= 0.003 AND best_epoch > 10.
  exploration_promote: holdout stable (>=0.50), clear won-feature effect
  (winner/loser gap > v06d15 gap), or Top200 insight guides next hypothesis.

rejection_evidence: >
  Train decisions < 5000, cross-check failure, notebook failure,
  holdout < 0.47, best_epoch <= 4.

go_no_go: go
