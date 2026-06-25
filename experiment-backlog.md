# Experiment Backlog

Track what's proposed, claimed, and completed.
Rules → `loop-harness.md`. Procedures → `loop-skills.md`.

**Before starting any experiment:** pull → claim → push (see loop-skills.md).

---

## Claimed

| Experiment | Tag | Machine | Date |
|---|---|---|---|
| — | — | — | — |

---

## Proposed

| # | Experiment | Tag | Notes |
|---|---|---|---|
| 1 | Fix Cell[25] SKIP_PIPELINE incompatibility: move `import_agent_from_source` to Cell[3]; re-run v07d4 offline PPO path to confirm `learning_promote` with smoke gate passing | conservative | Blocker for v07d4 `learning_promote`. Fast run (SKIP_PIPELINE=True, 8 offline epochs, ~30 min). |
| 2 | Add `torch.manual_seed(RANDOM_SEED)` to RL training; re-run 30-iter online PPO from v07d2 baseline to measure true expected winner_margin with reproducibility | conservative | v07d3 vs v07d2 variance (0.008 vs 0.034) is Torch RNG. Seed first before any HPO. |
| 3 | Fresh episode collection from v07d4 weights (warm-start); run K_OFFLINE=4 offline epochs | conservative | v07d3 episodes are behavioral=v07d3; v07d4 has drifted. Fresh data should improve offline training quality. |
| 4 | Speed up offline PPO inner loop: replace Python `for row in episodes` with numpy batch pre-concat to GPU | conservative | 8 offline epochs on 292k decisions takes ~30 min. Bottleneck is Python data loop, not GPU. |
| 5 | New RL algorithm: REINFORCE with moving-average baseline (replace PPO) | aggressive | PPO clip may suppress learning on stale episodes. REINFORCE has no clip — higher variance but no staleness penalty. |
| 6 | Feature expansion: add opponent-hand-size and energy-in-play signals to the 97-dim feature vector | aggressive | Current features may be missing key board-state signals. Feature dim change requires fresh episode collection. |
| 7 | Architecture change: option-wise attention (score each legal option against a board context vector) instead of flat MLP | aggressive | Flat MLP treats each option independently. Attention can model relative option quality. |
| 8 | Reward shaping: add dense intermediate reward based on prize-card differential change per step | aggressive | Current reward is sparse (prize cards at end). Dense shaping could accelerate RL convergence. |

---

## Completed (recent)

| Version | Machine | Type | Promotion | winner_margin | Notes |
|---|---|---|---|---|---|
| v0-07d4 | ei | conservative | exploration_promote | 0.057 | Offline PPO on v07d3 episodes (8 epochs). Smoke gate not confirmed (SKIP_PIPELINE + import_agent_from_source bug). |
| v0-07d3 | ei | conservative | exploration_promote | 0.008 | Episode saving infra validated. rl_episodes.pt (292k decisions, 1.1 GB). Torch seed variance (was 0.034 in v07d2). |
| v0-07d2 | ei | conservative | exploration_promote | 0.034 | Hybrid IL+RL (λ_il=1.0, λ_rl=0.05). First working RL version. Manual promotion. |
| v0-07d1 | ei | aggressive | reject | n/a | PPO RL only (no IL anchor). win_rate=0.375. winner_margin not measured. |
| v0-06d18 | ei | conservative | learning_promote | -0.004 | IL baseline (97-dim features + MLP reranker). Runtime baseline holdout_top1=0.509. |

---

## Research Baseline Reference

See `loop-harness.md` Baseline Handling for authoritative values.

- Current research baseline: `winner_margin = 0.057` (v07d4, stored-feature eval)
- Current runtime baseline: `holdout_model_top1 = 0.509` (guarded_torch_policy)
