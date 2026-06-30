# v0-07d1: PPO RL from IL Pre-training

## Goal
First RL experiment. Transition from imitation learning (IL) to reinforcement learning (RL).
Use v06d18 IL-trained model as initialization and fine-tune with PPO vs rule agent.

## Primary Metric
win_rate_vs_rule_agent: fraction of games won against rule agent (target > 0.5)

## Secondary Metrics
- winner_margin: winner_top1 - loser_top1 on v06d18 holdout (forgetting check)
- action_changes: must be > 0 (model is actually making different choices, guarded mode)

## Key Changes from v06d18
- Model: IL scorer → Actor-Critic (policy + value heads)
- Training: cross-entropy on replay → PPO on live game outcomes
- Deployment: shadow mode → guarded mode (action_changes > 0 required)
- Declared reuse: v06d18 feature matrix + decision rows (feature engineering unchanged)

## Hypothesis
IL-initialized RL policy can beat rule agent (win_rate > 0.5) after PPO fine-tuning.
KL regularization prevents catastrophic forgetting of IL pre-training.

## Self-Review (Bias Check)
- Risk: sparse rewards (only at game end) may be too weak for meaningful learning
- Risk: PPO with only 200 games/iter may have high variance
- Negative result: win_rate stays near 0.5 (unchanged from IL baseline) or drops below
- Forgetting indicator: winner_margin drops significantly below IL baseline
