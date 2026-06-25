## Meta-cognitive framing

Current state: v07d2 (learning_promote) winner_margin=+0.034 vs IL baseline -0.004. Total run time ~42 min (14 min pipeline + 28 min RL). RL game simulation accounts for 100% of RL cost (0.28s/game × 6000 games). PDCA cycle is bottlenecked by game simulation — each hyperparameter experiment re-simulates 6000 games unnecessarily.

Harness requirement for learning_promote: winner_margin >= 0.030. This is purely an infrastructure experiment; winner_margin should be ≈ v07d2's 0.034 (same algorithm, no model change).

Gap being closed: Enable RL episode data to be saved and reused across experiments. This is the prerequisite for v07d4+ to run in ~5-10 min instead of 42 min.

What's being tested: "Accumulating and saving episodes during the online training loop does not break the RL algorithm or change winner_margin materially."

Causal assumptions: torch.save of episode data is a write-only side effect with no impact on training. Assumption could be wrong if memory pressure from accumulating ~25M floats causes slowdowns or OOM.

Bias: Infrastructure experiments tempt easy "success" declarations. Counter: winner_margin must be within a reasonable range of v07d2's 0.034 (≥0.020) to count as validated.

Negative result: winner_margin drops below 0.020, or rl_episodes.pt is missing/corrupt, or OOM from episode accumulation.

---

## Version goal

```
version: pokemon-20260625-v0-07d3
baseline: see Baseline Handling section (research baseline winner_margin=0.034)
canonical_notebook: pokemon-20260625-v0-07d3.ipynb
development_track: exploration_track
runtime_mode: guarded_torch_policy
promotion_type_target: learning_promote
goal: Implement RL episode data separation — accumulate and save simulation episodes to
  rl_episodes.pt during training. Fix Cell[25] promotion criterion (winner_margin
  instead of rl_win_rate). Validate that winner_margin is maintained.
hypothesis: Adding episode saving (write-only side effect) and fixing the promotion
  criterion does not change winner_margin; the infrastructure is validated when
  winner_margin ≈ v07d2's 0.034.
change_scope:
  CHANGED: (1) Cell[23]: accumulate episodes per iter, torch.save at end.
           (2) Cell[25]: replace rl_win_rate>0.45 with winner_margin>0.030;
               add winner_margin/winner_top1/loser_top1/il_baseline fields to _promo.
           (3) Cell[1]: EXPERIMENT_NAME, RUN_PREFIX.
  UNCHANGED: RL algorithm, hyperparameters (λ_rl=0.05, λ_il=1.0, N_ITERS=30,
             GAMES_PER_ITER=200), feature engineering, model architecture,
             opponent mix, IL data source.
fresh_training: true
artifact_reuse_policy: v06d18 IL weights and feature matrix (declared reuse, same as v07d2)
episode_source: fresh (behavioral policy: v07d2 initialization — v06d18 IL weights)
datasets: v06d18 decision_rows.parquet + main_feature_matrix.npy (same as v07d2)
target_metric: winner_margin (holdout_winner_top1 − holdout_loser_top1)
baseline_winner_margin: 0.034 (v07d2 research baseline; see Baseline Handling)
success_criteria:
  - Notebook executes end-to-end.
  - rl_episodes.pt is written to OUTPUT_DIR and loadable.
  - winner_margin >= 0.020 (conservative lower bound; expect ≈ 0.034).
  - Promotion decision auto-generated as learning_promote (no manual correction).
  - Cell[25] uses winner_margin criterion (not rl_win_rate).
rollback_criteria:
  - winner_margin < 0.020 (indicates a bug introduced by episode accumulation).
  - rl_episodes.pt missing or OOM during accumulation.
  - Promotion decision still requires manual correction.
expected_artifacts:
  - pokemon-20260625-v0-07d3-rl_episodes.pt  (new)
  - pokemon-20260625-v0-07d3-v07d2_rl_report.json
  - pokemon-20260625-v0-07d3-main_hybrid_report.json
  - pokemon-20260625-v0-07d3-promotion-decision.json  (should auto-say learning_promote)
  - executed notebook
```
