"""Build pokemon-20260627-v0-08d11-remote-pc.ipynb from v08d1.

v08d11: bench Pokemon ID features — add bench slot IDs to state_summary.

Currently state_summary only knows bench COUNT (my_bench_count=2) but NOT
which Pokemon are on the bench. Adding individual bench slot IDs gives the
LGBM much richer board state information:
  - my_bench_id_0, my_bench_id_1, my_bench_id_2: my bench Pokemon IDs
  - op_bench_id_0, op_bench_id_1, op_bench_id_2: opponent bench Pokemon IDs

This requires re-mining episodes (no preload — new features need new extraction).
The run will take 2-3h for mining + ~30min for training.
After completion, the enhanced parquets can serve as preload for v08d12+.

Changes from v08d1 (not v08d2! Need full mining run):
  1. EXPERIMENT_NAME → v0_08d11_bench_ids
  2. RUN_PREFIX → pokemon-20260627-v0-08d11
  3. RUN_LOCAL_EVAL = False (keep fast)
  4. Cell[6]: Add bench slot IDs to state_summary_from_obs dict
  5. Cell[19]: Add bench_id features to UNKNOWN0_LGBM_CAT_OVERRIDES
     + modify _LGBM_INJ_CODE to extract bench IDs at inference time
  (No preload — this IS the new preload generator)
"""
import json

SRC = '/kaggle/working/pokemon-20260627-v0-08d1.ipynb'  # Build from v08d1 (has full mining)
DST = '/kaggle/working/pokemon-20260627-v0-08d11.ipynb'

with open(SRC) as f:
    nb = json.load(f)

cells = nb['cells']

# ── Cell[1]: Config ───────────────────────────────────────────────────────────
src1 = ''.join(cells[1]['source'])

OLD_EXP = "EXPERIMENT_NAME = 'v0_08d1_top200_lgbm_26'"
NEW_EXP = "EXPERIMENT_NAME = 'v0_08d11_bench_ids'"
assert OLD_EXP in src1, f'EXPERIMENT_NAME not found'
src1 = src1.replace(OLD_EXP, NEW_EXP)

OLD_PFX = "RUN_PREFIX = 'pokemon-20260627-v0-08d1'"
NEW_PFX = "RUN_PREFIX = 'pokemon-20260627-v0-08d11'"
assert OLD_PFX in src1, f'RUN_PREFIX not found'
src1 = src1.replace(OLD_PFX, NEW_PFX)

# Disable eval games for speed
OLD_LOCAL = 'RUN_LOCAL_EVAL = True'
NEW_LOCAL = 'RUN_LOCAL_EVAL = False   # v08d11: disabled for speed'
assert OLD_LOCAL in src1
src1 = src1.replace(OLD_LOCAL, NEW_LOCAL)

OLD_CONFIRM = 'RUN_CONFIRM_EVAL = True'
NEW_CONFIRM = 'RUN_CONFIRM_EVAL = False   # v08d11: disabled for speed'
assert OLD_CONFIRM in src1
src1 = src1.replace(OLD_CONFIRM, NEW_CONFIRM)

OLD_META = 'RUN_META_EVAL = True'
NEW_META = 'RUN_META_EVAL = False   # v08d11: disabled for speed'
assert OLD_META in src1
src1 = src1.replace(OLD_META, NEW_META)

cells[1]['source'] = src1.splitlines(keepends=True)

# ── Cell[6]: Add bench slot IDs to state_summary_from_obs ────────────────────
src6 = ''.join(cells[6]['source'])

OLD_BENCH = (
    "        'my_bench_count': len(my_bench),\n"
    "        'op_bench_count': len(op_bench),\n"
    "        'my_alakazam_count': sum(1 for x in [my_active] + my_bench if x.get('id') == Alakazam),"
)
NEW_BENCH = (
    "        'my_bench_count': len(my_bench),\n"
    "        'op_bench_count': len(op_bench),\n"
    "        # v08d11: bench slot IDs — which Pokemon is in each bench slot\n"
    "        'my_bench_id_0': (my_bench[0].get('id') or 0) if len(my_bench) > 0 else 0,\n"
    "        'my_bench_id_1': (my_bench[1].get('id') or 0) if len(my_bench) > 1 else 0,\n"
    "        'my_bench_id_2': (my_bench[2].get('id') or 0) if len(my_bench) > 2 else 0,\n"
    "        'op_bench_id_0': (op_bench[0].get('id') or 0) if len(op_bench) > 0 else 0,\n"
    "        'op_bench_id_1': (op_bench[1].get('id') or 0) if len(op_bench) > 1 else 0,\n"
    "        'op_bench_id_2': (op_bench[2].get('id') or 0) if len(op_bench) > 2 else 0,\n"
    "        'my_alakazam_count': sum(1 for x in [my_active] + my_bench if x.get('id') == Alakazam),"
)
assert OLD_BENCH in src6, 'bench_count not found in Cell[6]'
src6 = src6.replace(OLD_BENCH, NEW_BENCH)

cells[6]['source'] = src6.splitlines(keepends=True)

# ── Cell[19]: Add bench IDs to LGBM categorical features ─────────────────────
src19 = ''.join(cells[19]['source'])

# Add to UNKNOWN0_LGBM_CAT_OVERRIDES (card ID features treated as categorical)
OLD_CAT_OVR = "UNKNOWN0_LGBM_CAT_OVERRIDES = ["
# Find the full list to inject bench IDs after existing entries
cat_ovr_idx = src19.find(OLD_CAT_OVR)
assert cat_ovr_idx >= 0, 'LGBM_CAT_OVERRIDES not found'

# Find the closing ] of UNKNOWN0_LGBM_CAT_OVERRIDES
cat_end = src19.find(']', cat_ovr_idx)
OLD_CAT_BLOCK = src19[cat_ovr_idx:cat_end+1]
NEW_CAT_BLOCK = (
    OLD_CAT_BLOCK.rstrip(']')
    + "    'my_bench_id_0', 'my_bench_id_1', 'my_bench_id_2',\n"
    + "    'op_bench_id_0', 'op_bench_id_1', 'op_bench_id_2',\n"
    + "]"
)
src19 = src19.replace(OLD_CAT_BLOCK, NEW_CAT_BLOCK)

# Add bench IDs to the injection code: before rows=[]
OLD_INJ_ROWS = "rows = []\\n        for i, o in enumerate(select.option):"
NEW_INJ_ROWS = (
    "_my_bench_ids = [float(getattr(p, \\'id\\', 0) or 0) for p in my_bench]\\n"
    "        _op_bench_ids = [float(getattr(p, \\'id\\', 0) or 0) for p in op_bench]\\n"
    "        rows = []\\n        for i, o in enumerate(select.option):"
)
assert OLD_INJ_ROWS in src19, 'rows=[] in injection code not found'
src19 = src19.replace(OLD_INJ_ROWS, NEW_INJ_ROWS)

# Add bench IDs to each row dict (after option_signature)
OLD_INJ_ARCH = "   \\'option_signature\\': sig,\\n            })"
NEW_INJ_ARCH = (
    "   \\'option_signature\\': sig,\\n"
    "                \\'my_bench_id_0\\': _my_bench_ids[0] if len(_my_bench_ids) > 0 else 0.0,\\n"
    "                \\'my_bench_id_1\\': _my_bench_ids[1] if len(_my_bench_ids) > 1 else 0.0,\\n"
    "                \\'my_bench_id_2\\': _my_bench_ids[2] if len(_my_bench_ids) > 2 else 0.0,\\n"
    "                \\'op_bench_id_0\\': _op_bench_ids[0] if len(_op_bench_ids) > 0 else 0.0,\\n"
    "                \\'op_bench_id_1\\': _op_bench_ids[1] if len(_op_bench_ids) > 1 else 0.0,\\n"
    "                \\'op_bench_id_2\\': _op_bench_ids[2] if len(_op_bench_ids) > 2 else 0.0,\\n"
    "            })"
)
assert OLD_INJ_ARCH in src19, 'option_signature row dict closing not found'
src19 = src19.replace(OLD_INJ_ARCH, NEW_INJ_ARCH)

cells[19]['source'] = src19.splitlines(keepends=True)

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
s6 = ''.join(nb2['cells'][6]['source'])
s19 = ''.join(nb2['cells'][19]['source'])

assert 'v0_08d11_bench_ids' in s1,                  'EXPERIMENT_NAME'
assert "RUN_PREFIX = 'pokemon-20260627-v0-08d11'" in s1, 'RUN_PREFIX'
assert 'RUN_LOCAL_EVAL = False' in s1,               'eval disabled'
assert 'my_bench_id_0' in s6,                        'bench IDs in Cell[6]'
assert 'op_bench_id_2' in s6,                        'bench IDs in Cell[6]'
assert 'my_bench_id_0' in s19,                       'bench IDs in LGBM features'
assert '_my_bench_ids' in s19,                       'bench IDs in injection code'
assert '_op_bench_ids' in s19,                       'bench IDs in injection code'
full = ''.join(''.join(c['source']) for c in nb2['cells'])
assert 'lambdarank' in full, 'lambdarank kept'
print('All sanity checks passed.')
