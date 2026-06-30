"""Build pokemon-20260625-v0-07d8-remote-pc.ipynb from v07d7-remote-pc notebook.

v07d8: aggressive — online PPO + lambda_il=0.1 (extreme IL anchor reduction)
Changes vs v07d7:
  - Revert EPISODE_SOURCE: v07d6 episodes → None (online game collection)
  - Revert EPISODE_WARMSTART: v06d18 → None (start from IL init, no explicit warmstart)
  - lambda_il: 0.5 → 0.1 (very weak IL anchor; tests how far RL can dominate with PPO clip)
  - EXPERIMENT_NAME, RUN_PREFIX updated
  - Summary label updated
  Note: PPO_EPOCHS=4, VALUE_COEF=0.5, N_ITERS=50 unchanged
  Hypothesis: PPO clip provides stability even with very weak IL anchor (lambda_il=0.1);
              v07d7 showed PPO clip works at 0.5; this tests whether RL signal dominates
              further without catastrophic forgetting.
"""
import json

SRC = '/kaggle/working/pokemon-20260625-v0-07d7-remote-pc.ipynb'
DST = '/kaggle/working/pokemon-20260625-v0-07d8-remote-pc.ipynb'

with open(SRC) as f:
    nb = json.load(f)

cells = nb['cells']

# ── Cell[1]: Config ───────────────────────────────────────────────────────────
src1 = ''.join(cells[1]['source'])

src1 = src1.replace(
    "EXPERIMENT_NAME   = 'v0_07d7_offline_ppo_v07d6_episodes'",
    "EXPERIMENT_NAME   = 'v0_07d8_online_ppo_lambda_il_01'"
)
src1 = src1.replace(
    "RUN_PREFIX = os.environ.get('V07D7_RUN_PREFIX', 'pokemon-20260625-v0-07d7-remote-pc')",
    "RUN_PREFIX = os.environ.get('V07D8_RUN_PREFIX', 'pokemon-20260625-v0-07d8-remote-pc')"
)
# Revert EPISODE_SOURCE: v07d6 episodes → None
src1 = src1.replace(
    "EPISODE_SOURCE     = '/kaggle/working/pokemon-20260625-v0-07d6-remote-pc/pokemon-20260625-v0-07d6-remote-pc-rl_episodes.pt'  # v07d7: offline PPO on v07d6 buffer",
    "EPISODE_SOURCE     = None           # v07d8: online RL, fresh game collection"
)
# Revert EPISODE_WARMSTART: v06d18 → None
src1 = src1.replace(
    "EPISODE_WARMSTART  = '/kaggle/working/pokemon-20260624-v0-06d18/pokemon-20260624-v0-06d18-main_option_scorer.pt'  # v07d7: start from v06d18 IL",
    "EPISODE_WARMSTART  = None           # v07d8: start from IL init (online branch)"
)

assert 'v0_07d8_online_ppo_lambda_il_01' in src1, 'EXPERIMENT_NAME not updated'
assert 'v0-07d8-remote-pc' in src1, 'RUN_PREFIX not updated'
assert "EPISODE_SOURCE     = None" in src1, 'EPISODE_SOURCE not reverted'
assert "EPISODE_WARMSTART  = None" in src1, 'EPISODE_WARMSTART not reverted'
cells[1]['source'] = src1.splitlines(keepends=True)

# ── Cell[23]: RL training ─────────────────────────────────────────────────────
src23 = ''.join(cells[23]['source'])

# 1. lambda_il: 0.5 → 0.1
old_lambda = "    _V07D2_LAMBDA_IL    = 0.5  # v07d7: keep 0.5 (tests PPO clip vs REINFORCE forgetting)"
new_lambda = "    _V07D2_LAMBDA_IL    = 0.1  # v07d8: extreme reduction (tests PPO clip stability at very low IL anchor)"
assert old_lambda in src23, 'old lambda_il not found'
src23 = src23.replace(old_lambda, new_lambda)

# 2. Update summary label
src23 = src23.replace(
    "    print(f'\\n=== v0-07d7 Summary ===')",
    "    print(f'\\n=== v0-07d8 Summary ===')"
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
assert 'v0_07d8_online_ppo_lambda_il_01' in s1
assert 'v0-07d8-remote-pc' in s1
assert 'EPISODE_SOURCE     = None' in s1
assert 'EPISODE_WARMSTART  = None' in s1
assert '_V07D2_LAMBDA_IL    = 0.1' in s23
assert '_V07D2_PPO_EPOCHS   = 4' in s23
assert 'v0-07d8 Summary' in s23
print('All sanity checks passed.')
