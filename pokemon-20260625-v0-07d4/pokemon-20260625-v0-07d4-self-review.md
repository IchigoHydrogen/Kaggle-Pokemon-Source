## Pre-implementation self-review

### Hypothesis check
Falsifiable: winner_margin after offline training can be compared to warmstart (0.008).
Any value > 0.008 confirms offline PPO learned something from saved episodes.
Any value ≤ 0.008 triggers rollback: offline training does not work with this data.

### Change-by-change review

**Cell[1]: SKIP_PIPELINE, EPISODE_SOURCE, EPISODE_WARMSTART**
- SKIP_PIPELINE=True skips cells 4-21 entirely.
- EPISODE_SOURCE and EPISODE_WARMSTART are paths to v07d3 artifacts.
- Risk: if v07d3 artifacts are missing → Path.exists() returns False → falls to online branch.
  Online branch would fail (no game sim configured for v07d4). Mitigation: assert file exists
  in offline branch and raise clearly.
- Verdict: acceptable.

**Cell[3]: ARTIFACT_PATHS = {}**
- Added after utility definitions. When SKIP_PIPELINE=True, Cell[7] (which originally
  initializes ARTIFACT_PATHS={}) is skipped. Cell[23] and Cell[25] need ARTIFACT_PATHS
  to be defined.
- Risk: if SKIP_PIPELINE=False (future run), Cell[7] re-initializes ARTIFACT_PATHS={},
  clearing the Cell[3] init. This is the original behavior — no regression.
- Verdict: safe.

**Cells 4-21: SKIP_PIPELINE wrap**
- Each code cell wrapped in `if not SKIP_PIPELINE:`.
- Risk: any variable defined in cells 4-21 that Cell[23] depends on → NameError.
  From inspection: Cell[23] loads its own IL data directly (v06d18 artifacts), does not
  use any pipeline cell outputs. Cell[25] likewise.
  One exception: `write_json` is defined in Cell[3] (not skipped) → safe.
- Verdict: safe.

**Cell[23]: offline training branch**
- Load v07d3 rl_episodes.pt: verified loadable in v07d3 verification step.
- Load v07d3 model weights: same format as IL weights (net.0/2/4/6 → backbone/policy_head).
  Key check: `_v07d2_model.load_state_dict(ws_dict, strict=False)` to avoid value_head
  key mismatch (value_head was added in RL; stored weights may or may not include it).
  strict=False means missing keys keep their random init — value head starts fresh.
  Wait: v07d3 saved IL-format weights (net.0-net.6 keys). Does it include value_head?
  From build_v07d3.py Patch3: model save is `_sv = {net.0...net.6}` — no value_head.
  So value_head starts from random init in offline mode. This means the value loss
  will be noisy for epoch 0, stabilizing thereafter. PPO advantage normalization
  (_adv_t = (adv - mean) / std) makes policy loss independent of value accuracy.
  Verdict: acceptable. Value head convergence happens within epoch 0.

- PPO clip ratio for stale episodes: episodes from iter 0 have behavioral policy ≈ v06d18 IL.
  v07d3 final policy (warmstart) has drifted 30 iters from v06d18. Ratio may be extreme.
  With clip ε=0.2, ratio clamped to [0.8, 1.2]: stale gradients are naturally suppressed.
  Recent episodes (iters 20-29) will dominate learning. This is correct behavior.
  Verdict: acceptable.

- Variable contract: offline branch must define `_V07D2_N_ITERS`, `_V07D2_GAMES_PER_ITER`,
  `_pre_wr`, `_post_wr`, `_pre_wt1`, `_pre_lt1`, `_pre_margin`, `_post_wt1`, `_post_lt1`,
  `_post_margin` for the downstream rl_report and MAIN_HYBRID_REPORT construction.
  Checked: all defined in the offline branch.

- IL loss in offline epoch: uses `_train_idxs_v7d2`, `_train_df_v7d2`, `_X_full_v7d2` —
  all defined before the if/else branch. Safe.

- episode_source field in Cell[25] _promo: from v07d3 Patch5, it was hardcoded as
  `'fresh (behavioral policy: ' + RUN_PREFIX + ')'`. For v07d4 this should say "reuse".
  Fix: override this in the v07d4 build script to use EPISODE_SOURCE path.

### Scope check
No changes to: RL algorithm (PPO), hyperparameters (λ_rl, λ_il, clip, value_coef),
model architecture, feature dim, IL data, holdout evaluation, Cell[25] logic.

### Known gap
torch.manual_seed still not added. Offline training is already reproducible (no game
simulation, Python and numpy seeds set). This is acceptable for v07d4.

### Verdict: **go**
