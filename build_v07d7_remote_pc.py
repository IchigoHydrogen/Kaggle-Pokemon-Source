"""Build pokemon-20260625-v0-07d7-remote-pc.ipynb from v07d6-remote-pc notebook.

v07d7: conservative — offline PPO on v07d6 episode buffer (481k decisions)
Changes vs v07d6:
  - EPISODE_SOURCE: None → v07d6 rl_episodes.pt (offline mode, no game collection)
  - EPISODE_WARMSTART: None → v06d18 IL model (clean IL starting point)
  - Restore VALUE_COEF: 0.1 → 0.5, PPO_EPOCHS: 1 → 4 (online branch cleanup)
  - lambda_il: keep 0.5 (tests whether PPO clip prevents forgetting that REINFORCE showed)
  - Add inference-feature eval to offline branch output
  - N_ITERS default stays at 50 (but offline uses _K_OFFLINE=8 epochs, not N_ITERS)
  - EXPERIMENT_NAME, RUN_PREFIX updated
"""
import json

SRC = '/kaggle/working/pokemon-20260625-v0-07d6-remote-pc.ipynb'
DST = '/kaggle/working/pokemon-20260625-v0-07d7-remote-pc.ipynb'

with open(SRC) as f:
    nb = json.load(f)

cells = nb['cells']

# ── Cell[1]: Config ───────────────────────────────────────────────────────────
src1 = ''.join(cells[1]['source'])

src1 = src1.replace(
    "EXPERIMENT_NAME   = 'v0_07d6_reinforce_lower_lambda_il'",
    "EXPERIMENT_NAME   = 'v0_07d7_offline_ppo_v07d6_episodes'"
)
src1 = src1.replace(
    "RUN_PREFIX = os.environ.get('V07D6_RUN_PREFIX', 'pokemon-20260625-v0-07d6-remote-pc')",
    "RUN_PREFIX = os.environ.get('V07D7_RUN_PREFIX', 'pokemon-20260625-v0-07d7-remote-pc')"
)
# EPISODE_SOURCE: None → v07d6 episodes
src1 = src1.replace(
    "EPISODE_SOURCE     = None           # v07d5: online RL, no pre-saved episodes",
    "EPISODE_SOURCE     = '/kaggle/working/pokemon-20260625-v0-07d6-remote-pc/pokemon-20260625-v0-07d6-remote-pc-rl_episodes.pt'  # v07d7: offline PPO on v07d6 buffer"
)
# EPISODE_WARMSTART: None → v06d18 IL model
src1 = src1.replace(
    "EPISODE_WARMSTART  = None           # v07d5: IL initialization only",
    "EPISODE_WARMSTART  = '/kaggle/working/pokemon-20260624-v0-06d18/pokemon-20260624-v0-06d18-main_option_scorer.pt'  # v07d7: start from v06d18 IL"
)

assert 'v0_07d7_offline_ppo_v07d6_episodes' in src1, 'EXPERIMENT_NAME not updated'
assert 'v0-07d7-remote-pc' in src1, 'RUN_PREFIX not updated'
assert 'rl_episodes.pt' in src1, 'EPISODE_SOURCE not updated'
assert 'v0-06d18-main_option_scorer.pt' in src1, 'EPISODE_WARMSTART not updated'
cells[1]['source'] = src1.splitlines(keepends=True)

# ── Cell[23]: RL training ─────────────────────────────────────────────────────
src23 = ''.join(cells[23]['source'])

# 1. Restore PPO hyperparams (online branch cleanup: CLIP comment, VALUE_COEF, PPO_EPOCHS)
old_hparams = (
    "    _V07D2_CLIP         = 0.2  # not used (REINFORCE)\n"
    "    _V07D2_VALUE_COEF   = 0.1  # reduced: value baseline for GAE only\n"
    "    _V07D2_ENTROPY_COEF = 0.01\n"
    "    _V07D2_LAMBDA_RL    = 0.05\n"
    "    _V07D2_LAMBDA_IL    = 0.5  # v07d6: reduced to let RL compete\n"
    "    _V07D2_PPO_EPOCHS   = 1    # v07d6: single pass (REINFORCE, no stale-lp issue)"
)
new_hparams = (
    "    _V07D2_CLIP         = 0.2\n"
    "    _V07D2_VALUE_COEF   = 0.5\n"
    "    _V07D2_ENTROPY_COEF = 0.01\n"
    "    _V07D2_LAMBDA_RL    = 0.05\n"
    "    _V07D2_LAMBDA_IL    = 0.5  # v07d7: keep 0.5 (tests PPO clip vs REINFORCE forgetting)\n"
    "    _V07D2_PPO_EPOCHS   = 4"
)
assert old_hparams in src23, "old_hparams not found"
src23 = src23.replace(old_hparams, new_hparams)

# 2. Add inference eval after offline branch final eval (after line 756 area)
old_offline_final = (
    "        print(f'Offline final: wt1={_post_wt1:.4f} lt1={_post_lt1:.4f} margin={_post_margin:.4f}')\n"
    "        print(f'vs warmstart:  margin_delta={_post_margin - _pre_margin:+.4f}')"
)
new_offline_final = (
    "        print(f'Offline final: wt1={_post_wt1:.4f} lt1={_post_lt1:.4f} margin={_post_margin:.4f}')\n"
    "        print(f'vs warmstart:  margin_delta={_post_margin - _pre_margin:+.4f}')\n"
    "        _inf_wt1, _inf_lt1, _inf_margin = _v07d2_winner_margin_inference(\n"
    "            _v07d2_model, _X_full_v7d2, _holdout_df_v7d2, _v07d2_device)\n"
    "        print(f'Offline inference-feature: wm={_inf_margin:.4f} '\n"
    "              f'(wt1={_inf_wt1:.4f} lt1={_inf_lt1:.4f}) delta_vs_stored={_inf_margin-_post_margin:+.4f}')"
)
assert old_offline_final in src23, "old_offline_final not found"
src23 = src23.replace(old_offline_final, new_offline_final)

# 3. Update summary label
src23 = src23.replace(
    "    print(f'\\n=== v0-07d6 Summary ===')",
    "    print(f'\\n=== v0-07d7 Summary ===')"
)

cells[23]['source'] = src23.splitlines(keepends=True)

# ── Clear outputs ─────────────────────────────────────────────────────────────
for c in cells:
    c['outputs'] = []
    c['execution_count'] = None

with open(DST, 'w') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f'Written: {DST}')

# Sanity checks
with open(DST) as f:
    nb2 = json.load(f)
s1 = ''.join(nb2['cells'][1]['source'])
s23 = ''.join(nb2['cells'][23]['source'])
assert 'v0_07d7_offline_ppo_v07d6_episodes' in s1
assert 'v0-07d7-remote-pc' in s1
assert 'rl_episodes.pt' in s1
assert 'v0-06d18-main_option_scorer.pt' in s1
assert '_V07D2_LAMBDA_IL    = 0.5' in s23
assert '_V07D2_PPO_EPOCHS   = 4' in s23
assert 'Offline inference-feature' in s23
assert 'v0-07d7 Summary' in s23
print('All sanity checks passed.')
