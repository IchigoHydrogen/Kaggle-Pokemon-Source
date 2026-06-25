"""Build pokemon-20260625-v0-07d6-remote-pc.ipynb from v07d5-remote-pc notebook.

Changes vs v07d5:
  - REINFORCE (no PPO clip): replace importance-ratio clip with direct -log_prob * adv
  - lambda_il: 1.0 -> 0.5  (let RL compete with IL anchor)
  - PPO_EPOCHS: 4 -> 1     (single pass; no stale-log-prob issue without importance sampling)
  - VALUE_COEF: 0.5 -> 0.1 (value baseline still used for GAE but less heavily trained)
  - N_ITERS: 30 -> 50      (RTX4090 fast: 30 iters in 31 min, 50 in ~52 min)
  - EXPERIMENT_NAME, RUN_PREFIX updated
"""
import json

SRC = '/kaggle/working/pokemon-20260625-v0-07d5-remote-pc.ipynb'
DST = '/kaggle/working/pokemon-20260625-v0-07d6-remote-pc.ipynb'

with open(SRC) as f:
    nb = json.load(f)

cells = nb['cells']

# ── Cell[1]: Config ───────────────────────────────────────────────────────────
src1 = ''.join(cells[1]['source'])

src1 = src1.replace(
    "EXPERIMENT_NAME   = 'v0_07d5_full_pipeline_new_data'",
    "EXPERIMENT_NAME   = 'v0_07d6_reinforce_lower_lambda_il'"
)
src1 = src1.replace(
    "RUN_PREFIX = os.environ.get('V07D5_RUN_PREFIX', 'pokemon-20260625-v0-07d5-remote-pc')",
    "RUN_PREFIX = os.environ.get('V07D6_RUN_PREFIX', 'pokemon-20260625-v0-07d6-remote-pc')"
)

cells[1]['source'] = src1.splitlines(keepends=True)

# ── Cell[23]: RL training ─────────────────────────────────────────────────────
src23 = ''.join(cells[23]['source'])

# 1. Hyperparams: lambda_il 1.0->0.5, PPO_EPOCHS 4->1, VALUE_COEF 0.5->0.1
old_hparams = (
    "    _V07D2_CLIP         = 0.2\n"
    "    _V07D2_VALUE_COEF   = 0.5\n"
    "    _V07D2_ENTROPY_COEF = 0.01\n"
    "    _V07D2_LAMBDA_RL    = 0.05\n"
    "    _V07D2_LAMBDA_IL    = 1.0\n"
    "    _V07D2_PPO_EPOCHS   = 4"
)
new_hparams = (
    "    _V07D2_CLIP         = 0.2  # not used (REINFORCE)\n"
    "    _V07D2_VALUE_COEF   = 0.1  # reduced: value baseline for GAE only\n"
    "    _V07D2_ENTROPY_COEF = 0.01\n"
    "    _V07D2_LAMBDA_RL    = 0.05\n"
    "    _V07D2_LAMBDA_IL    = 0.5  # v07d6: reduced to let RL compete\n"
    "    _V07D2_PPO_EPOCHS   = 1    # v07d6: single pass (REINFORCE, no stale-lp issue)"
)
assert old_hparams in src23, "old_hparams not found"
src23 = src23.replace(old_hparams, new_hparams)

# 2. Replace PPO clip with REINFORCE in _v07d2_update
# Old: ratio computation + clip
old_ppo_clip = (
    "                    ratio = torch.exp(lp_i[ro['actions'][i]] - old_lp[i])\n"
    "                    ai    = adv_t[i]\n"
    "                    pl    = -torch.min(ratio * ai,\n"
    "                                       torch.clamp(ratio, 1 - _V07D2_CLIP, 1 + _V07D2_CLIP) * ai)"
)
new_reinforce = (
    "                    lp_a  = lp_i[ro['actions'][i]]  # fresh log-prob (REINFORCE)\n"
    "                    ai    = adv_t[i]\n"
    "                    pl    = -lp_a * ai               # no importance ratio, no clip"
)
assert old_ppo_clip in src23, "old_ppo_clip not found"
src23 = src23.replace(old_ppo_clip, new_reinforce)

# 3. N_ITERS default: 30 -> 50 in full mode
old_niters = (
    "            _V07D2_N_ITERS        = int(os.environ.get('V07D2_N_ITERS',        '30'))\n"
    "            _V07D2_GAMES_PER_ITER = int(os.environ.get('V07D2_GAMES_PER_ITER', '200'))"
)
new_niters = (
    "            _V07D2_N_ITERS        = int(os.environ.get('V07D2_N_ITERS',        '50'))\n"
    "            _V07D2_GAMES_PER_ITER = int(os.environ.get('V07D2_GAMES_PER_ITER', '200'))"
)
assert old_niters in src23, "old_niters not found"
src23 = src23.replace(old_niters, new_niters)

# 4. Update summary label
src23 = src23.replace(
    "    print(f'\\n=== v0-07d5 Summary ===')",
    "    print(f'\\n=== v0-07d6 Summary ===')"
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
assert 'v0_07d6_reinforce_lower_lambda_il' in s1
assert 'v0-07d6-remote-pc' in s1
assert '_V07D2_LAMBDA_IL    = 0.5' in s23
assert '_V07D2_PPO_EPOCHS   = 1' in s23
assert 'no importance ratio, no clip' in s23
assert "V07D2_N_ITERS',        '50'" in s23
print('All sanity checks passed.')
