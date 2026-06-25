# Experiment Backlog

Track what's proposed, claimed, and completed.
Rules → `loop-harness.md`. Procedures → `loop-skills.md`.

**Before starting any experiment:** pull → claim → push (see loop-skills.md).

---

## Claimed

| Experiment | Tag | Machine | Date |
|---|---|---|---|
| (none) | | | |

---

## Proposed

| # | Experiment | Tag | Notes |
|---|---|---|---|
| 1 | Offline PPO on v07d6 episode buffer (481k decisions); inference-feature eval primary metric | conservative | v07d6 generated 481k decisions. Test offline PPO on that buffer to see if PPO learns better than online REINFORCE. |
| 2 | Online PPO + reduce lambda_il to 0.3 (keep clip); monitor forgetting vs learning | conservative | v07d6 showed REINFORCE (no clip) + lambda_il=0.5 = forgetting. Hypothesis: PPO clip prevents large updates. lambda_il=0.3 further reduces anchor while retaining stability. |
| 3 | Remove dim96 from feature vector; retrain IL from scratch on 96-dim features | aggressive | dim96 is win-label leakage. Dropping it forces model to learn genuine board features. IL retraining required. |
| 4 | Online PPO + lambda_il=0.1 (extreme reduction); measure how much RL dominates before forgetting | aggressive | Tests lower bound of IL anchor with PPO clip retained. Measures REINFORCE-vs-PPO tradeoff in forgetting regime. |
| 5 | Feature expansion: add opponent-hand-size and energy-in-play signals to the 97-dim feature vector | aggressive | Current features may be missing key board-state signals. Feature dim change requires fresh episode collection. |
| 6 | Architecture change: option-wise attention instead of flat MLP | aggressive | Flat MLP treats each option independently. Attention can model relative option quality. |
| 7 | Reward shaping: add dense intermediate reward based on prize-card differential change per step | aggressive | Current reward is sparse. Dense shaping could accelerate RL convergence. |

---

## Completed (recent)

| Version | Machine | Type | Promotion | winner_margin (stored) | winner_margin (inference) | Notes |
|---|---|---|---|---|---|---|
| v0-07d6-remote-pc | remote-pc | aggressive | reject | -0.0147 | -0.0186 | REINFORCE+lambda_il=0.5: forgetting (IL base=-0.0037 → post-RL=-0.0147). High-variance REINFORCE + weak anchor drove policy away from IL. Gates pass. 25 min (50 iters, smoke env var didn't reach kernel). |
| v0-07d5-remote-pc | remote-pc | conservative | exploration_promote | 0.0068 | -0.0009 | CRITICAL: dim96 leakage confirmed. Inference margin near-zero. All prior stored margins inflated. Smoke gate confirmed. 31 min on RTX4090. |
| v0-07d4 | ei | conservative | exploration_promote | 0.057 | n/a (not measured) | Offline PPO on v07d3 episodes (8 epochs). Stored margin inflated by dim96. Smoke gate not confirmed. |
| v0-07d3 | ei | conservative | exploration_promote | 0.008 | n/a | Episode saving infra. Torch seed variance. |
| v0-07d2 | ei | conservative | exploration_promote | 0.034 | n/a | Hybrid IL+RL first working version. |
| v0-07d1 | ei | aggressive | reject | n/a | n/a | PPO only (no IL anchor). |
| v0-06d18 | ei | conservative | learning_promote | -0.004 | n/a | IL baseline. Runtime baseline holdout_top1=0.509. |

---

## Research Baseline Reference

See `loop-harness.md` Baseline Handling for authoritative values.

- **Stored-feature eval (inflated — do not use for new comparisons):** `winner_margin = 0.057` (v07d4)
- **Inference-feature eval (authoritative):** `winner_margin ≈ -0.001` (v07d5-remote-pc; near-zero)
- Current runtime baseline: `holdout_model_top1 = 0.509` (guarded_torch_policy)
