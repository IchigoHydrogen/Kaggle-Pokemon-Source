"""Build pokemon-20260625-v0-07d14-remote-pc.ipynb from v07d13-remote-pc notebook.

v07d14: conservative — extend N_ITERS from 100 to 150.

Finding from v07d13: with 100 iters, the best checkpoint is at iter 99 (the final
iteration), meaning the model is still improving when training stops. Extending to
150 iters may reveal a higher peak. This is conservative (50 more iters vs +100 for
an aggressive jump to 200).

Trajectory from v07d13 (iter: wm):
  0:-0.0059 10:-0.0056 20:-0.0066 30:-0.0161 40:-0.0073
  50:+0.0209 60:-0.0047 70:+0.0153 80:+0.0093 90:+0.0037 99:+0.0269

iter-99 = +0.0269, still trending up. 50 more iters should be informative.

No changes to hyperparameters, architecture, or training algorithm.
"""
import json

SRC = '/kaggle/working/pokemon-20260625-v0-07d13-remote-pc.ipynb'
DST = '/kaggle/working/pokemon-20260625-v0-07d14-remote-pc.ipynb'

with open(SRC) as f:
    nb = json.load(f)

cells = nb['cells']

# ── Cell[1]: Config ───────────────────────────────────────────────────────────
src1 = ''.join(cells[1]['source'])

src1 = src1.replace(
    "EXPERIMENT_NAME   = 'v0_07d13_n_iters_100'",
    "EXPERIMENT_NAME   = 'v0_07d14_n_iters_150'"
)
src1 = src1.replace(
    "RUN_PREFIX = os.environ.get('V07D13_RUN_PREFIX', 'pokemon-20260625-v0-07d13-remote-pc')",
    "RUN_PREFIX = os.environ.get('V07D14_RUN_PREFIX', 'pokemon-20260625-v0-07d14-remote-pc')"
)
assert 'v0_07d14_n_iters_150' in src1, 'EXPERIMENT_NAME'
assert 'v0-07d14-remote-pc' in src1,   'RUN_PREFIX'
cells[1]['source'] = src1.splitlines(keepends=True)

# ── Cell[23]: RL training ─────────────────────────────────────────────────────
src23 = ''.join(cells[23]['source'])

# Extend N_ITERS: 100 → 150
OLD_NITERS = "        _V07D2_N_ITERS        = int(os.environ.get('V07D2_N_ITERS',        '100'))  # v07d13: extended from 50"
NEW_NITERS = "        _V07D2_N_ITERS        = int(os.environ.get('V07D2_N_ITERS',        '150'))  # v07d14: extended from 100"
assert OLD_NITERS in src23, 'N_ITERS=100 not found'
src23 = src23.replace(OLD_NITERS, NEW_NITERS)

# Update summary label
assert "print(f'\\n=== v0-07d13 Summary ===')" in src23, 'summary label'
src23 = src23.replace(
    "    print(f'\\n=== v0-07d13 Summary ===')",
    "    print(f'\\n=== v0-07d14 Summary ===')"
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

assert 'v0_07d14_n_iters_150' in s1,              'EXPERIMENT_NAME'
assert 'v0-07d14-remote-pc' in s1,                'RUN_PREFIX'
assert '_V07D2_FEAT_DIM      = 96' in s23,         'FEAT_DIM=96'
assert "'150'))" in s23,                            'N_ITERS=150 default'
assert '_v07d2_best_margin' in s23,                'best ckpt tracking'
assert 'v07d12 saved model: wm=' in s23,           'measurement fix kept'
assert '_sv97' in s23,                             '97-dim padding kept'
assert 'v0-07d14 Summary' in s23,                  'summary label'
assert '_V07D2_LAMBDA_IL    = 0.5' in s23,         'lambda_il=0.5'
print('All sanity checks passed.')
