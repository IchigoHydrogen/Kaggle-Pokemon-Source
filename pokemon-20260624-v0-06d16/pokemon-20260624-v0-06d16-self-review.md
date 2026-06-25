# Pre-Implementation Self-Review: v0-06d16

experiment_plan: >
  Replace uniform CE loss with MC-return-weighted CE (winner=1, loser=0). Keep
  v06d15 features (96-dim card metadata), architecture, deck, and deterministic
  split unchanged. Train fresh model. Compare holdout top-1 to v06d15 (0.5074)
  and v06d11 model baseline (0.509). Report winner/loser breakdown separately.

implementation_plan: >
  Copy v06d15 notebook to pokemon-20260624-v0-06d16.ipynb. In Cell 0, update
  EXPERIMENT_NAME and RUN_PREFIX. In Cell 19, after building _tr_chosen, add
  _tr_won = _train_df['won'].astype(float).values.astype(np.float32). In the
  training batch loop, replace single F.cross_entropy call with:
    _b_won = _tr_won[_bidxs]
    _w_tensor = torch.tensor(_b_won, ..., device=_device)
    _per_loss = F.cross_entropy(_logits, _chosen_t, reduction='none')
    _w_sum = _w_tensor.sum().clamp(min=1.0)
    _loss = (_per_loss * _w_tensor).sum() / _w_sum
  Add mc_return_report JSON (winner_frac, effective_train_size, holdout winner/loser top-1).

why_this_is_the_next_best_step: >
  v06d15 confirmed PLAY as primary next target via winner-conditioned diagnostics.
  Before adding PLAY-specific runtime logic, the minimal useful question is: does
  outcome-weighting in the loss alone improve holdout top-1? This is one isolated
  change that directly tests the MC return signal without new features or runtime risk.

what_would_make_this_result_untrustworthy:
  - Winner fraction in train deviates far from 0.50 (biased deck/matchup distribution
    means winner-only is not a fair sample of all game situations).
  - Holdout improvement is < 0.003 (within run-to-run noise).
  - Winner-side holdout top-1 improves but loser-side degrades badly, suggesting
    the model shifts its distribution rather than learning better play.
  - Training loss hits exactly 0.0 in early batches (numeric instability with
    very small w_sum).

expected_failure_modes:
  - Winner fraction is ~0.50 but the model already converges to same solution
    (null result: MC return does not help with this data and architecture).
  - Effective training batch size is smaller per step (only winner decisions
    contribute gradient), so learning is slower; may need more epochs.
  - Early stopping triggers earlier on valid, underfitting relative to v06d15.
  - Loss normalization by w_sum causes instability in batches with very few winners.

scope_guardrails:
  - No feature dimension changes (must remain 96-dim, same layout as v06d15).
  - No architecture changes.
  - No deck changes.
  - No runtime action changes (shadow mode, action_changes must = 0).
  - No holdout tuning.
  - No ATTACK, ABILITY, or END-specific logic in this version.
  - Exactly one loss variant: winner-only MC return (w in {0, 1}).
  - Do not add a second loss variant (0.5 loser weight) to this version.

validation_plan: >
  Execute v06d16 notebook end-to-end. Verify feature cross-check passes. Check
  that winner fraction in train is between 0.40 and 0.60. Compare holdout top-1
  to v06d15 (0.5074) and v06d11 (0.509). Report separate winner-side vs loser-side
  holdout top-1. Run shadow runtime probe: verify action_changes=0, illegal=0,
  exceptions=0. Write promotion decision.

promotion_evidence_required: >
  learning_promote: holdout top-1 > v06d15 (0.5074) by >= 0.003 AND winner-side
  holdout top-1 improves relative to v06d15 winner-side accuracy.
  exploration_promote: holdout top-1 stable (>= 0.50) with clear MC-return impact
  evidence (winner/loser accuracy divergence), enabling next hypothesis.

rejection_evidence: >
  Holdout top-1 < 0.47, feature cross-check failure, notebook execution failure,
  effective winner train count < 5000, or training loss = 0 from first batch.

go_no_go: go
