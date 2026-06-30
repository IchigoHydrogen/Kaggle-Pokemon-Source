## Meta-cognitive framing

Current state: v07d3 (exploration_promote) winner_margin=+0.008 vs IL baseline -0.004.
Infrastructure validated: rl_episodes.pt (292,276 decisions, 1.1 GB) written and loadable.
Total run time: 38 min (14 min pipeline + 24 min RL game simulation).
PDCA bottleneck is game simulation (Python engine, 97% of RL cost).

Gap being closed: Use the saved rl_episodes.pt to run offline PPO without game simulation.
If this works, future RL experiments run in ~5-10 min instead of 38 min.

What's being tested: "Offline PPO epochs on v07d3's saved episodes improve winner_margin
without any new game simulation. Run time drops to ~10 min."

Causal assumptions:
- Warmstarting from v07d3 final weights gives a good starting point.
- PPO clip naturally handles stale episodes (early-iter episodes are down-weighted by clipping).
- GAE advantages stored in v07d3 episodes are off-policy-compatible with the current policy.
- IL anchor (λ_il=1.0) remains active during offline epochs to prevent forgetting.

Potential failure modes:
- Offline PPO clips all gradients to zero (policy too far from behavioral) → no learning.
- Value function mismatch causes instability → winner_margin oscillates or regresses.
- IL loss conflicts with PPO gradient direction → winner_margin does not improve.

Bias check: Episode reuse is the session's strategic bet. Counter: rollback is defined
clearly — if winner_margin < 0.008 (v07d3 result), offline training is not helping.

---

## Version goal

```
version: pokemon-20260625-v0-07d4
baseline: see Baseline Handling section (research baseline winner_margin=0.034)
canonical_notebook: pokemon-20260625-v0-07d4.ipynb
development_track: exploration_track
runtime_mode: guarded_torch_policy
promotion_type_target: learning_promote
goal: Validate offline PPO episode reuse. Load v07d3's rl_episodes.pt, warm-start from
  v07d3's final model weights, and run 8 offline PPO epochs without game simulation.
  Also implement SKIP_PIPELINE=True to cut pipeline time from 14 min to 0.
hypothesis: Offline PPO on v07d3's saved episodes (292,276 decisions) will drive
  winner_margin above v07d3's 0.008, validated by holdout eval after each epoch.
  Total run time drops to ~10 min.
change_scope:
  CHANGED: (1) Cell[1]: add SKIP_PIPELINE, EPISODE_SOURCE, EPISODE_WARMSTART; bump
               RUN_PREFIX/EXPERIMENT_NAME.
           (2) Cell[3]: add ARTIFACT_PATHS = {} init after utilities.
           (3) Cells 4-21: wrap code cells in `if not SKIP_PIPELINE:`.
           (4) Cell[23]: add offline training branch (load episodes, run K_OFFLINE=8
               PPO epochs, skip game simulation). Online branch preserved for reuse.
  UNCHANGED: PPO hyperparameters (λ_rl=0.05, λ_il=1.0, clip=0.2), model architecture,
             feature engineering, IL data source, Cell[25] promotion logic.
fresh_training: false
artifact_reuse_policy: v07d3 rl_episodes.pt + v07d3 final model weights (declared reuse)
episode_source: reuse (behavioral policy: pokemon-20260625-v0-07d3)
datasets: v06d18 decision_rows.parquet + main_feature_matrix.npy (IL anchor, same as v07d3)
target_metric: winner_margin (holdout_winner_top1 − holdout_loser_top1)
baseline_winner_margin: 0.034 (v07d2 research baseline)
success_criteria:
  - Notebook executes end-to-end in under 20 min total.
  - winner_margin improves from warmstart 0.008 during offline epochs.
  - winner_margin > 0.008 after offline training (any improvement from warmstart counts).
  - SKIP_PIPELINE=True path works (no NameError from write_json/ARTIFACT_PATHS).
rollback_criteria:
  - winner_margin <= 0.008 after all offline epochs (no learning from saved episodes).
  - Any NameError or missing-variable error from pipeline skip.
expected_artifacts:
  - pokemon-20260625-v0-07d4-main_option_scorer.pt
  - pokemon-20260625-v0-07d4-v07d2_rl_report.json
  - pokemon-20260625-v0-07d4-main_hybrid_report.json
  - pokemon-20260625-v0-07d4-promotion-decision.json
  - executed notebook (total < 20 min)
```
