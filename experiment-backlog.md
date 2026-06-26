# Experiment Backlog

Track what's proposed, claimed, and completed.
Rules → `loop-harness.md`. Procedures → `loop-skills.md`.

**Before starting any experiment:** pull → claim → push (see loop-skills.md).

---

## Claimed

_(none)_

---

## Proposed

| # | Experiment | Tag | Notes |
|---|---|---|---|
| 1 | Online PPO + lambda_il=0.3 (midpoint), start from v07d7 model as warmstart | conservative | v07d7 (0.5)=learning_promote, v07d8 (0.1)=reject. Test midpoint. |
| 2 | Online PPO + reduce lambda_il to 0.3 (keep clip), start from v07d7 | conservative | Moderate anchor reduction. Safer than 0.1. |
| 3 | Remove dim96 from feature vector; retrain IL from scratch on 96-dim features | aggressive | dim96 is win-label leakage. Dropping it forces model to learn genuine board features. IL retraining required. |
| 4 | Collect fresh episodes with v07d7 model, offline PPO 8-16 epochs | aggressive | Better behavioral policy → better episode distribution. Tests if fresh data improves offline PPO further. |
| 5 | Feature expansion: add opponent-hand-size and energy-in-play signals to the 97-dim feature vector | aggressive | Current features may be missing key board-state signals. Feature dim change requires fresh episode collection. |
| 6 | Architecture change: option-wise attention instead of flat MLP | aggressive | Flat MLP treats each option independently. Attention can model relative option quality. |
| 7 | Reward shaping: add dense intermediate reward based on prize-card differential change per step | aggressive | Current reward is sparse. Dense shaping could accelerate RL convergence. |

---

## Completed (recent)

| Version | Machine | Type | Promotion | winner_margin (stored) | winner_margin (inference) | Notes |
|---|---|---|---|---|---|---|
| v0-07d15-remote-pc | remote-pc | conservative | exploration_promote | +0.0219 | +0.0219 | Memory fix validated (peak 43.7GB vs v07d13 46GB). Speed: 92.4min vs 89.7min — gc.collect/iter slow. wm=+0.0219 < baseline +0.0269 (run variance). Key: 150-iter runs now feasible (~59GB est). Remove gc.collect/iter in v07d16. |
| v0-07d14-remote-pc | remote-pc | conservative | **needs_followup** | — | — | FAILED: DeadKernelError (OOM) during Cell[23]. Memory: ~29GB→62GB+ over 150 iters. Docker hit RAM ceiling. Pipeline (cells 1-22) completed; rl_episodes.pt=0 bytes; no model produced. Research baseline unchanged (+0.0269). Root cause: episode buffer grows O(n_iters). Fix: cap buffer or reduce iters. |
| v0-07d13-remote-pc | remote-pc | aggressive | **learning_promote** | +0.0269 | **+0.0269** | N_ITERS=100. iter-99 = best-ckpt (+0.0269). Model still converging at final iter — needs more iters. Massive gain: 50 iters peak=+0.0080 → 100 iters peak=+0.0269 (+0.0189 improvement). Part B PASS. New research baseline=+0.0269. |
| v0-07d12-remote-pc | remote-pc | conservative | exploration_promote | +0.0077 | +0.0077 | Measurement fix confirmed. iter-40 best-ckpt=+0.0077. Part B pass. Below v07d11 (+0.0080) by 0.0003 (run variance). Research baseline stays at +0.0080. |
| v0-07d11-remote-pc | remote-pc | aggressive | **learning_promote** | +0.0080 | +0.0046 (bug) | dim96 removal + online PPO + best-ckpt. Part B PASS. Best-ckpt restored iter40=+0.0080 > research baseline +0.0059. rl_report shows +0.0046 (measurement timing bug: computed before restore). Actual saved model=iter40 (+0.0080). New research baseline=+0.0080. Fix measurement in v07d12. |
| v0-07d10-remote-pc | remote-pc | aggressive | exploration_promote | — | -0.000044 | dim96 removal working: n_valid=200/iter, 42.7 min. IL baseline -0.0059 → final -0.000044 (+0.0059 improvement). Peak iter40=+0.0096 > research baseline. Hard gate FAIL: Part B size mismatch (96-dim model vs 97-dim class). Fix: re-save as 97-dim (zero pad). Also: no best-ckpt saving → final<peak. Fix both in v07d11. |
| v0-07d9-remote-pc | remote-pc | aggressive | needs_followup | — | -0.0059 (IL only) | dim96 removal: 2 bugs. Bug1: live_features hardcoded np.zeros((n,97)) → shape mismatch → 0 games collected. Bug2: Part B main.py used 97-dim class. 96-dim IL baseline = -0.0059 (stored=inference). Fix both in v07d10. |
| v0-07d8-remote-pc | remote-pc | aggressive | reject | 0.0046 | -0.0012 | Online PPO + lambda_il=0.1: too weak IL anchor. Inference wm -0.0012 < v07d5 baseline -0.0009. Policy drifted from IL without winner-selection benefit. Win rate 0.25→0.29 (game-level but not winner-selective). 43 min RL. |
| v0-07d7-remote-pc | remote-pc | conservative | **learning_promote** | 0.0388 | **+0.0059** | Offline PPO (8 epochs x 481k decisions from v07d6). PPO clip prevented forgetting (REINFORCE failed). inference wm +0.0059 > baseline -0.0009 (+0.0068 improvement). All gates pass. 90 min total (65 min RL). |
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
- **Inference-feature eval (authoritative):** `winner_margin = +0.0269` (v07d13-remote-pc; learning_promote; iter-99 best-ckpt; prev: +0.0080 v07d11)
- Current runtime baseline: `holdout_model_top1 = 0.509` (guarded_torch_policy)
