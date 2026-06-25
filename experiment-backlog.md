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
| 1 | REINFORCE with moving-average baseline (replace PPO) + reduce lambda_il to 0.5 | aggressive | v07d5 showed win_rate collapse 0.59→0.26 under PPO+lambda_il=1.0. REINFORCE has no clip; larger updates. Primary metric: inference-feature winner_margin. |
| 2 | Offline PPO on large fresh episode buffer with inference-feature eval as primary metric | conservative | Re-establish calibrated baseline. v07d4 0.057 was inflated. Need offline PPO with dim96 leakage removed to get true episodic learning signal. |
| 3 | Reduce lambda_il from 1.0 to 0.1 in online RL; monitor win_rate recovery | conservative | lambda_il=1.0 suppresses RL signal too strongly. Lower anchor allows RL to drive actual game wins. |
| 4 | Remove dim96 from feature vector; retrain IL from scratch | aggressive | dim96 is unavailable at true inference (win-label leakage). Dropping it removes the inflation source and forces the model to learn from genuine board features. |
| 5 | Feature expansion: add opponent-hand-size and energy-in-play signals to the 97-dim feature vector | aggressive | Current features may be missing key board-state signals. Feature dim change requires fresh episode collection. |
| 6 | Architecture change: option-wise attention (score each legal option against a board context vector) instead of flat MLP | aggressive | Flat MLP treats each option independently. Attention can model relative option quality. |
| 7 | Reward shaping: add dense intermediate reward based on prize-card differential change per step | aggressive | Current reward is sparse (prize cards at end). Dense shaping could accelerate RL convergence. |

---

## Completed (recent)

| Version | Machine | Type | Promotion | winner_margin (stored) | winner_margin (inference) | Notes |
|---|---|---|---|---|---|---|
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
