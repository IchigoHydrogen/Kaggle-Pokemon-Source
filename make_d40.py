"""Create v05d40 from v05d36 (0.0601):
num_leaves 63 → 127, keeping d36's 5 engineered features.

d36: 5 new features, leaves=63, best_iter=106 → good convergence, best=0.0601
d39: 9 new features, leaves=63, best_iter=12  → overcrowded, marginal gain 0.0603

d36 best_iter=106 means the model was still learning deeply with leaves=63.
More leaves = more complex splits. With leaves=127, we should be able to capture
interactions like (can_attack × option_type), (hp_diff × option_signature), etc.

127 = 2^7 - 1. min_data_in_leaf stays at 20 (default) to avoid overfit.
"""
import json, uuid
from pathlib import Path

WORK_DIR = Path('/kaggle/working')
SRC_NB = WORK_DIR / 'pokemon-20260625-v0-05d36.ipynb'  # base: d36 (best clean)
DST_NB = WORK_DIR / 'pokemon-20260625-v0-05d40.ipynb'

with open(SRC_NB) as f:
    nb = json.load(f)

def cell_src(i):
    return ''.join(nb['cells'][i]['source'])

def set_cell_src(i, s):
    nb['cells'][i]['source'] = s
    nb['cells'][i]['outputs'] = []
    if nb['cells'][i].get('cell_type') == 'code':
        nb['cells'][i]['execution_count'] = None

# ── Cell 1 ────────────────────────────────────────────────────────────────────
src1 = cell_src(1)
src1 = src1.replace("EXPERIMENT_NAME = 'v0_05d36_lgbm_feateng'",
                    "EXPERIMENT_NAME = 'v0_05d40_lgbm_leaves127_feateng'")
src1 = src1.replace("RUN_PREFIX = 'pokemon-20260625-v0-05d36'",
                    "RUN_PREFIX = 'pokemon-20260625-v0-05d40'")
set_cell_src(1, src1)
assert 'v0_05d40_lgbm_leaves127_feateng' in cell_src(1)
print("Cell 1: OK")

cell19_idx = None
for i, cell in enumerate(nb['cells']):
    src = ''.join(cell['source'])
    if 'lambdarank' in src and 'UNKNOWN0_MLP_REPORT' in src:
        cell19_idx = i
        break
assert cell19_idx is not None
src19 = cell_src(cell19_idx)

# ── Patch 1: num_leaves 63 → 127 ─────────────────────────────────────────────
OLD_LEAVES = "                    'num_leaves': 63,"
NEW_LEAVES = "                    'num_leaves': 127,"
assert OLD_LEAVES in src19, "OLD_LEAVES not found!"
src19 = src19.replace(OLD_LEAVES, NEW_LEAVES)
assert "'num_leaves': 127" in src19

# ── Patch 2: model name ───────────────────────────────────────────────────────
OLD_MODEL_NAME = "                    'model': 'lightgbm_lambdarank_feateng',"
NEW_MODEL_NAME = "                    'model': 'lightgbm_lambdarank_leaves127_feateng',"
assert OLD_MODEL_NAME in src19
src19 = src19.replace(OLD_MODEL_NAME, NEW_MODEL_NAME)

set_cell_src(cell19_idx, src19)
c19 = cell_src(cell19_idx)
assert "'num_leaves': 127" in c19 and 'leaves127_feateng' in c19 and 'can_attack' in c19
print("Cell 19: OK (num_leaves=127 + d36 features)")

for cell in nb['cells']:
    if 'id' not in cell:
        cell['id'] = str(uuid.uuid4())[:8]
for cell in nb['cells']:
    if cell.get('cell_type') == 'code':
        cell['outputs'] = []
        cell['execution_count'] = None

with open(DST_NB, 'w') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print(f"\nv05d40 notebook written: {DST_NB}")
