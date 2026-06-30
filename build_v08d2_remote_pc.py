"""Build pokemon-20260627-v0-08d2-remote-pc.ipynb from v08d1.

v08d2: same top200 data + LGBM LambdaRank, but eval games disabled.

Goal: Fast iteration mode (~45-60 min instead of 5-6h for v08d1).
Remove local eval + confirm eval + meta eval to get rapid results.
Use NDCG@1 from LGBM training and holdout metrics for evaluation instead.

Changes from v08d1:
  1. EXPERIMENT_NAME → v0_08d2_top200_lgbm_noeval
  2. RUN_PREFIX → pokemon-20260627-v0-08d2
  3. RUN_LOCAL_EVAL = False
  4. RUN_CONFIRM_EVAL = False
  5. RUN_META_EVAL = False
"""
import json

SRC = '/kaggle/working/pokemon-20260627-v0-08d1.ipynb'
DST = '/kaggle/working/pokemon-20260627-v0-08d2.ipynb'

with open(SRC) as f:
    nb = json.load(f)

cells = nb['cells']

# ── Cell[1]: Config ───────────────────────────────────────────────────────────
src1 = ''.join(cells[1]['source'])

# Patch 1: EXPERIMENT_NAME
OLD_EXP = "EXPERIMENT_NAME = 'v0_08d1_top200_lgbm_26'"
NEW_EXP = "EXPERIMENT_NAME = 'v0_08d2_top200_lgbm_noeval'"
assert OLD_EXP in src1, f'EXPERIMENT_NAME not found: {OLD_EXP!r}'
src1 = src1.replace(OLD_EXP, NEW_EXP)

# Patch 2: RUN_PREFIX
OLD_PFX = "RUN_PREFIX = 'pokemon-20260627-v0-08d1'"
NEW_PFX = "RUN_PREFIX = 'pokemon-20260627-v0-08d2'"
assert OLD_PFX in src1, f'RUN_PREFIX not found'
src1 = src1.replace(OLD_PFX, NEW_PFX)

# Patch 3: Disable eval games
OLD_LOCAL = 'RUN_LOCAL_EVAL = True'
NEW_LOCAL = 'RUN_LOCAL_EVAL = False   # v08d2: disabled for speed'
assert OLD_LOCAL in src1, f'RUN_LOCAL_EVAL not found'
src1 = src1.replace(OLD_LOCAL, NEW_LOCAL)

OLD_CONFIRM = 'RUN_CONFIRM_EVAL = True'
NEW_CONFIRM = 'RUN_CONFIRM_EVAL = False   # v08d2: disabled for speed'
assert OLD_CONFIRM in src1, f'RUN_CONFIRM_EVAL not found'
src1 = src1.replace(OLD_CONFIRM, NEW_CONFIRM)

OLD_META = 'RUN_META_EVAL = True'
NEW_META = 'RUN_META_EVAL = False   # v08d2: disabled for speed'
assert OLD_META in src1, f'RUN_META_EVAL not found'
src1 = src1.replace(OLD_META, NEW_META)

cells[1]['source'] = src1.splitlines(keepends=True)

# ── Clear outputs ─────────────────────────────────────────────────────────────
for c in cells:
    if c.get('cell_type') == 'code':
        c['outputs'] = []
        c['execution_count'] = None
    else:
        c.pop('outputs', None)
        c.pop('execution_count', None)

with open(DST, 'w') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f'Written: {DST}')

# ── Sanity checks ─────────────────────────────────────────────────────────────
with open(DST) as f:
    nb2 = json.load(f)
s1 = ''.join(nb2['cells'][1]['source'])

assert 'v0_08d2_top200_lgbm_noeval' in s1,   'EXPERIMENT_NAME'
assert "RUN_PREFIX = 'pokemon-20260627-v0-08d2'" in s1, 'RUN_PREFIX'
assert 'RUN_LOCAL_EVAL = False' in s1,         'RUN_LOCAL_EVAL disabled'
assert 'RUN_CONFIRM_EVAL = False' in s1,        'RUN_CONFIRM_EVAL disabled'
assert 'RUN_META_EVAL = False' in s1,           'RUN_META_EVAL disabled'
assert 'episodes-2026-06-26' in s1,             'episode dir 20260626 kept'
assert 'top200-20260626-ranking' in s1,          'ranking 20260626 kept'

full = ''.join(''.join(c['source']) for c in nb2['cells'])
assert 'lambdarank' in full,    'lambdarank kept'
assert 'winner_weight' in full, 'winner_weight kept'
print('All sanity checks passed.')
