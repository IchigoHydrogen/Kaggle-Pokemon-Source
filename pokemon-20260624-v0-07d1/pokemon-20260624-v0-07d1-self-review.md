# v0-07d1 Self-Review

## What I am testing
Can PPO RL fine-tuning of an IL-initialized policy improve win rate against rule agent?

## What I am NOT testing
- Multi-opponent generalization (only rule agent)
- Deck variety (same deck both sides)
- Reward shaping (only sparse terminal reward)

## Causal assumptions
1. IL model has good enough representations to generalize to RL setting
2. Game outcomes correlate with decision quality (true but noisy)
3. KL regularization coefficient (0.05) is appropriate
4. 200 games/iter provides sufficient signal for PPO to update meaningfully

## Where assumptions could fail
1. IL policy might be good imitation but poor RL initialization (distribution shift)
2. Opponents being the same (rule agent) limits diversity of training signal
3. KL coeff too high → model stays near IL, never learns RL signal
4. KL coeff too low → catastrophic forgetting within a few iterations

## How to distinguish positive from negative result
- Positive: win_rate > 0.5 AND winner_margin >= IL baseline (no forgetting)
- Marginal: win_rate ~ 0.5 AND margin preserved → IL at least not hurt
- Negative: win_rate < 0.5 OR margin drops significantly below IL baseline

## What comes next
- v0-07d2: Reward shaping, curriculum (weak → strong opponents), diverse seeds
- v0-07d3: IMPALA-style distributed collection if single-thread is bottleneck
