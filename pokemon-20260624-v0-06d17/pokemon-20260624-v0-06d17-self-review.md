# Pre-Implementation Self-Review: v0-06d17

experiment_plan: >
  Change loser weight in MC return from 0.0 to 0.5. Winner=1.0 / loser=0.5.
  Keep all other settings identical to v06d16. Compare holdout top-1 and
  best_epoch to diagnose whether the v06d16 underfitting was purely due to
  gradient starvation from zero-weighting losers.

implementation_plan: >
  Copy v06d16 notebook to v06d17. Update EXPERIMENT_NAME and RUN_PREFIX in
  Cell 0. In Cell 19, change exactly one line:
    FROM: _w_tensor = torch.tensor(_b_won, dtype=torch.float32, device=_device)
    TO:   _w_tensor = torch.tensor(0.5 + 0.5 * _b_won, dtype=torch.float32, device=_device)
  Update mc_return_report loss field to 'mc_return_soft_winner_1_loser_0p5'.
  No other changes.

why_this_is_the_next_best_step: >
  v06d16 proved that pure winner-only filtering underfits (epoch 4 vs 20 in
  v06d15). The softest possible fix is to restore 50% of the loser gradient.
  This isolates the underfitting cause and tests whether soft weighting is
  sufficient to recover accuracy while retaining winner-bias.

what_would_make_this_result_untrustworthy:
  - best_epoch still <= 4 (soft weighting not enough to fix underfitting).
  - holdout top-1 == v06d15 (0.5074) exactly (no effect from weighting).
  - winner/loser gap disappears (model treats them equally again).
  - Improvement is within noise (< 0.003 delta from v06d15).

expected_failure_modes:
  - Soft weighting is still too asymmetric; model converges to same solution
    as uniform CE but slower (null result).
  - 0.5 is not the right loser weight; optimal is somewhere between 0 and 0.5.
  - The underfitting in v06d16 had another cause (e.g., batch normalization or
    AdamW momentum), not just gradient starvation.

scope_guardrails:
  - Exactly one parameter change: loser weight 0.0 → 0.5.
  - No feature changes (96-dim, identical to v06d15/v06d16).
  - No architecture changes.
  - No deck changes.
  - No runtime action changes (shadow, action_changes must = 0).
  - No holdout tuning.

validation_plan: >
  Execute v06d17 notebook end-to-end. Check: feature cross-check passed,
  best_epoch > 4, holdout top-1 vs v06d15 and v06d16. Report winner/loser
  holdout top-1 breakdown. Shadow probe zero illegal/exceptions/action_changes.

promotion_evidence_required: >
  learning_promote: holdout >= v06d15 (0.5074) AND best_epoch > 10.
  exploration_promote: holdout stable (>= 0.50), best_epoch > 4, winner/loser
  gap persists — confirms soft weighting direction is correct for follow-up.

rejection_evidence: >
  Holdout < 0.47, feature mismatch, notebook failure, best_epoch <= 4.

go_no_go: go
