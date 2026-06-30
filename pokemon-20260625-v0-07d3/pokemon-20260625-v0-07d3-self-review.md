## Pre-implementation self-review

### Hypothesis check
The hypothesis is falsifiable: if winner_margin stays ≥0.020, episode saving is a neutral side effect.
If it drops below 0.020, there's a bug (likely memory pressure or an unintended mutation of the rollout dict during accumulation).

### Change-by-change review

**Cell[23]: episode buffer accumulation**
- Location: inside the main RL loop, after `_adv, _ret = _v07d2_gae(_ro)`
- Action: `extend` lists with data from `_ro`. These are the same Python objects the update step already consumed, so no mutation risk.
- Memory estimate: 30 iters × 200 games × ~45 decisions × 97 floats × 4 bytes ≈ 100 MB. Fine for the container.
- After loop: `torch.save` dict to `OUTPUT_DIR / f'{RUN_PREFIX}-rl_episodes.pt'`. One-time write, does not affect training.
- Risk: None identified.

**Cell[25]: promotion criterion**
- Old: `if _rl_win_rate > 0.50 → rl_promote; elif > 0.45 → needs_followup; else → reject`
- New: `if _all_gates and _quality_ok and _winner_margin > 0.030 → learning_promote`
- `quality_ok` is already set in Cell[23] as `bool(_post_margin > 0.030)`, so the logic is consistent.
- Also add winner_margin / winner_top1 / loser_top1 / il_baseline fields to `_promo` dict; these come from MAIN_HYBRID_REPORT and MAIN_LEARNING_REPORT which are already available.
- Risk: None. This is a pure criterion fix. The manual correction in v07d2 documented this exact change.

**Cell[1]: EXPERIMENT_NAME, RUN_PREFIX**
- Mechanical bump. No risk.

### Scope check
No changes to: RL algorithm, hyperparameters, feature engineering, model architecture, opponent mix, IL data source, Cell[25] smoke/archive/loader logic.

### Verdict: **go**
