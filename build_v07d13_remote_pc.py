"""Build pokemon-20260625-v0-07d13-remote-pc.ipynb from v07d12-remote-pc notebook.

v07d13: aggressive — extend N_ITERS from 50 to 100.

Hypothesis: iter-40 is consistently the best checkpoint with 50 iters (v07d11: +0.0080,
v07d12: +0.0077). The policy is still improving at iter 40 and declines toward iter 49.
Running 100 iters may reveal a higher peak beyond iter 40, or confirm that iter 40 is
the global optimum for this setup. Best-checkpoint saving will capture whichever iter
is best across all 100 iters.

No changes to hyperparameters, architecture, or training algorithm.
"""
import json

SRC = '/kaggle/working/pokemon-20260625-v0-07d12-remote-pc.ipynb'
DST = '/kaggle/working/pokemon-20260625-v0-07d13-remote-pc.ipynb'

with open(SRC) as f:
    nb = json.load(f)

cells = nb['cells']

# ── Cell[1]: Config ───────────────────────────────────────────────────────────
src1 = ''.join(cells[1]['source'])

src1 = src1.replace(
    "EXPERIMENT_NAME   = 'v0_07d12_measurement_fix'",
    "EXPERIMENT_NAME   = 'v0_07d13_n_iters_100'"
)
src1 = src1.replace(
    "RUN_PREFIX = os.environ.get('V07D12_RUN_PREFIX', 'pokemon-20260625-v0-07d12-remote-pc')",
    "RUN_PREFIX = os.environ.get('V07D13_RUN_PREFIX', 'pokemon-20260625-v0-07d13-remote-pc')"
)
assert 'v0_07d13_n_iters_100' in src1, 'EXPERIMENT_NAME'
assert 'v0-07d13-remote-pc' in src1,   'RUN_PREFIX'
cells[1]['source'] = src1.splitlines(keepends=True)

# ── Cell[23]: RL training ─────────────────────────────────────────────────────
src23 = ''.join(cells[23]['source'])

# Extend N_ITERS: 50 → 100 (online PPO default from env var)
OLD_NITERS = "        _V07D2_N_ITERS        = int(os.environ.get('V07D2_N_ITERS',        '50'))"
NEW_NITERS = "        _V07D2_N_ITERS        = int(os.environ.get('V07D2_N_ITERS',        '100'))  # v07d13: extended from 50"
assert OLD_NITERS in src23, 'N_ITERS default 50 not found'
src23 = src23.replace(OLD_NITERS, NEW_NITERS)

# Update summary label
assert "print(f'\\n=== v0-07d12 Summary ===')" in src23, 'summary label'
src23 = src23.replace(
    "    print(f'\\n=== v0-07d12 Summary ===')",
    "    print(f'\\n=== v0-07d13 Summary ===')"
)

cells[23]['source'] = src23.splitlines(keepends=True)

# ── Clear outputs ─────────────────────────────────────────────────────────────
for c in cells:
    c['outputs'] = []
    c['execution_count'] = None

with open(DST, 'w') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f'Written: {DST}')

# ── Sanity checks ─────────────────────────────────────────────────────────────
with open(DST) as f:
    nb2 = json.load(f)
s1  = ''.join(nb2['cells'][1]['source'])
s23 = ''.join(nb2['cells'][23]['source'])

assert 'v0_07d13_n_iters_100' in s1,              'EXPERIMENT_NAME'
assert 'v0-07d13-remote-pc' in s1,                'RUN_PREFIX'
assert '_V07D2_FEAT_DIM      = 96' in s23,         'FEAT_DIM=96'
assert "'100'))" in s23,                            'N_ITERS=100 default'
assert '_v07d2_best_margin' in s23,                'best ckpt tracking'
assert 'v07d12 saved model: wm=' in s23,           'measurement fix kept'
assert '_sv97' in s23,                             '97-dim padding kept'
assert 'v0-07d13 Summary' in s23,                  'summary label'
assert '_V07D2_LAMBDA_IL    = 0.5' in s23,         'lambda_il=0.5'
print('All sanity checks passed.')
