version: pokemon-20260625-v0-07d2
baseline: pokemon-20260624-v0-07d1 (reject; winner_margin=-0.0006)
research_baseline: pokemon-20260624-v0-06d18 (learning_promote; holdout winner_margin=0.050)
canonical_notebook: pokemon-20260625-v0-07d2.ipynb
development_track: exploration_track
runtime_mode: guarded_torch_policy
promotion_type_target: learning_promote

goal: >
  Fix all structural failures identified in v07d1 simultaneously.
  v07d1 showed RL fine-tuning collapsed winner_margin (0.050 → -0.0006) because:
  (1) IL training data was abandoned during RL; (2) KL penalty alone could not prevent
  catastrophic forgetting; (3) rule-agent-only opponent caused over-specialization;
  (4) sparse terminal reward made credit assignment hard.
  v07d2 applies four simultaneous fixes: IL-anchored hybrid loss, prize step rewards,
  diverse opponents, and removal of the standalone KL penalty.

hypothesis: >
  Hybrid IL+RL training (λ_il=1.0, λ_rl=0.05) with prize step rewards (±0.3),
  diverse opponents (35% rule / 30% IL / 35% self-play), and no KL penalty
  can maintain winner_margin ≥ IL baseline (0.050) while also improving
  win rate against diverse opponents, thereby closing all v07d1 failure modes.

harness_note: >
  The harness "1 experiment 1 hypothesis" rule is explicitly suspended here per
  user agreement (2026-06-25). All four changes address the SAME root cause
  (training objective misalignment with evaluation metric) and cannot be tested
  independently because each alone is insufficient to recover winner_margin.

change_scope: >
  Cell [23] rewrite only:
    - IL loss added to PPO update (λ_il=1.0, λ_rl=0.05)
    - Prize step rewards (±0.3 per prize delta)
    - Diverse opponent selection per game (35/30/35%)
    - KL penalty removed
  Cell [01]: EXPERIMENT_NAME, RUN_PREFIX
  Cell [25]: comment update only (same guarded probe logic)

fresh_training: true (IL weights reloaded from v06d18 fresh; no RL fine-tuned weights reused)
artifact_reuse_policy: >
  Reuse v06d18 feature matrix + decision rows (feature engineering unchanged, declared).
  No reuse of v07d1 model weights.

datasets: v06d18 replay/log data (same fixed split: train/valid/holdout)

target_metric: winner_margin (holdout_winner_top1 − holdout_loser_top1)
baseline_winner_margin: 0.050 (v06d18 research baseline, Top200 holdout)

success_criteria:
  - Notebook executes end-to-end
  - holdout winner_margin ≥ 0.030 (60% of IL baseline — accepts slight degradation from RL)
  - winner_margin does NOT collapse to near-zero (v07d1 level: -0.0006)
  - win rate vs diverse opponents ≥ pre-RL baseline
  - action_changes > 0 in guarded runtime probe
  - Zero illegal actions, zero exceptions in smoke eval

rollback_criteria:
  - winner_margin ≤ 0 (same collapse as v07d1)
  - winner_margin < 0.010 (trivially above zero but no signal)
  - Notebook execution fails

expected_artifacts:
  - executed notebook
  - v07d2_rl_report.json (winner_margin, win_rate, rl_history with per-iter wm)
  - main_option_scorer.pt
  - promotion-decision.json
  - goal.md (this file)
  - self-review.md
  - submission archives (guarded mode)
