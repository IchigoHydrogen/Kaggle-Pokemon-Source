# v0-07d2 Pre-Implementation Self-Review

## experiment_plan
Fix all v07d1 structural failures via Hybrid IL+RL:
  1. IL loss (λ=1.0) as dominant anchor alongside PPO (λ=0.05)
  2. Prize step rewards (±0.3 per prize delta) for denser credit assignment
  3. Diverse opponents: 35% rule / 30% v06d18 IL policy / 35% self-play
  4. KL penalty removed (IL loss is stronger and cleaner anchor)

## implementation_plan
- Copy v07d1 notebook → pokemon-20260625-v0-07d2.ipynb
- Rewrite Cell [23]: hybrid collect() + ppo_il_update() + diverse opponent functions
- Cell [01]: EXPERIMENT_NAME + RUN_PREFIX
- Run notebook end-to-end in container (GPU, ~35-45 min estimated)
- Evaluate winner_margin on holdout at training end + every 10 iters

## why_this_is_the_next_best_step
v07d1 demonstrated that pure RL fine-tuning completely destroys the IL signal
(winner_margin -0.0006). Returning to pure IL would be safe but won't advance
toward the long-term RL goal. This version is the minimum fix set needed to
make RL training non-destructive. Without all four changes, at least one
failure mode persists.

## what_would_make_this_result_untrustworthy
- If winner_margin check at iter 0 already differs from IL pre-training baseline,
  the IL loss isn't loading correctly → abort
- If prize delta is always 0, step rewards add no signal (inspect rl_history)
- If self-play generates degenerate games (very long or all errors), the self-play
  signal is noise → reduce self-play fraction in v07d3

## expected_failure_modes
A. IL loss and RL loss gradient directions still conflict → winner_margin stays low
   (λ_il=1.0 should dominate, but λ_rl=0.05 may still create tension)
B. Prize step rewards are always 0 (prize field extraction fails / game ends before prizes)
   → step rewards are effectively disabled, reduces to v07d1-like signal
C. Self-play opponents too weak early → degenerate game distributions
   → self-play games provide noisy gradients for both IL and RL losses
D. Training is slower (IL batch per step) and doesn't converge in 30 iters
   → increase N_ITERS in v07d3

## scope_guardrails
- Feature engineering unchanged from v06d18 (declared reuse)
- Model architecture unchanged from v07d1 (same AC)
- NO hyperparameter search: all λ values fixed at planned levels
- NO new feature dimensions: feat_dim stays 97
- If any single change appears to be a dominant failure, record it for v07d3
  (do NOT re-run with different hyperparameters mid-execution)

## validation_plan
Primary: winner_margin on fixed holdout (956 decisions) at end of training
  - Compare to IL baseline (0.050) and v07d1 result (-0.0006)
Secondary: per-iter winner_margin monitored every 10 iters
  - If winner_margin is declining by iter 10, IL anchor is failing
Win rate: tracked per iteration vs diverse opponents (not a promotion criterion)
Smoke: 176 games vs random agent (legality + crash check only)
Runtime: guarded probe (action_changes > 0 check)

## promotion_evidence_required
- winner_margin ≥ 0.030 on holdout
- winner_margin is STABLE or IMPROVING across iters (not declining like v07d1 KL divergence)
- Zero illegal actions in smoke eval
- action_changes > 0 in guarded mode

## rejection_evidence
- winner_margin ≤ 0 (same collapse as v07d1) → reject
- winner_margin between 0 and 0.010 (recovers partially but no meaningful signal) → reject
- winner_margin declining monotonically with training iters → reject (explore_promote at best)

## causal_assumptions
1. IL loss at λ_il=1.0 dominates PPO gradient and prevents forgetting
   Risk: if PPO loss scale is large, 0.05× still causes drift
   Mitigation: monitor winner_margin at iter 10 checkpoint
2. Prize changes are detectable from obs['current']['players'][seat]['prize']
   Risk: prize field structure may differ from _v06d15_player_summary
   Mitigation: if prize_delta is always 0, step rewards are 0 (safe fallback)
3. Self-play snapshot at iter start is a valid training signal
   Risk: early self-play model is weak → near-random opponent signal
   Mitigation: only 35% of games use self-play; rule and IL cover 65%

## go_no_go
GO — all failure modes have plausible mitigations, and even a partial fix
(winner_margin recovering from -0.0006 to any positive value) is informative
evidence for v07d3 direction.
