"""Create v05d34 from v05d25:
Larger model: num_leaves 63 → 255.

d25 uses num_leaves=63, best_iter=161. The model converges at 161 iterations with
only 63 leaves — suggesting the model capacity is the binding constraint.
Doubling/quadrupling leaves allows more complex decision boundary learning.

Also increase min_child_samples to 30 (from default 20) to prevent overfit with
more leaves. Keep winner_weight=4x and max_rows=350k.

Why 255 specifically: 2^8 - 1, can represent very complex interactions.
With 350k training rows and early_stopping=50, overfit risk is controlled.
"""
import json, uuid
from pathlib import Path

WORK_DIR = Path('/kaggle/working')
SRC_NB = WORK_DIR / 'pokemon-20260625-v0-05d25.ipynb'
DST_NB = WORK_DIR / 'pokemon-20260625-v0-05d34.ipynb'

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
src1 = src1.replace("EXPERIMENT_NAME = 'v0_05d25_lgbm_winner_wt4'",
                    "EXPERIMENT_NAME = 'v0_05d34_lgbm_leaves255'")
src1 = src1.replace("RUN_PREFIX = 'pokemon-20260625-v0-05d25'",
                    "RUN_PREFIX = 'pokemon-20260625-v0-05d34'")
set_cell_src(1, src1)
assert 'v0_05d34_lgbm_leaves255' in cell_src(1)
print("Cell 1: OK")

cell19_idx = None
for i, cell in enumerate(nb['cells']):
    src = ''.join(cell['source'])
    if 'lambdarank' in src and 'UNKNOWN0_MLP_REPORT' in src:
        cell19_idx = i
        break
assert cell19_idx is not None
src19 = cell_src(cell19_idx)

# ── Patch 1: num_leaves 63 → 255, min_child_samples 20 → 30 ──────────────────
OLD_LEAVES = "                    'num_leaves': 63,"
NEW_LEAVES = "                    'num_leaves': 255,"
assert OLD_LEAVES in src19, "OLD_LEAVES not found!"
src19 = src19.replace(OLD_LEAVES, NEW_LEAVES)

# Add min_child_samples after num_leaves (prevents overfit with large trees)
OLD_AFTER = (
    "                    'num_leaves': 255,\n"
    "                    'min_data_in_leaf': 20,"
)
# Check if min_data_in_leaf already exists
if 'min_data_in_leaf' in src19:
    OLD_MDIL = "                    'min_data_in_leaf': 20,"
    NEW_MDIL = "                    'min_data_in_leaf': 30,"
    assert OLD_MDIL in src19, f"OLD_MDIL not found! Searching: {OLD_MDIL}"
    src19 = src19.replace(OLD_MDIL, NEW_MDIL)
    print("min_data_in_leaf updated: 20 → 30")
else:
    # Insert min_data_in_leaf after num_leaves line
    OLD_NL_LINE = "                    'num_leaves': 255,"
    NEW_NL_LINE = "                    'num_leaves': 255,\n                    'min_data_in_leaf': 30,"
    src19 = src19.replace(OLD_NL_LINE, NEW_NL_LINE)
    print("min_data_in_leaf=30 inserted after num_leaves")

# ── Patch 2: model name ───────────────────────────────────────────────────────
OLD_MODEL_NAME = "                    'model': 'lightgbm_lambdarank_winner_wt4',"
NEW_MODEL_NAME = "                    'model': 'lightgbm_lambdarank_leaves255',"
assert OLD_MODEL_NAME in src19, "OLD_MODEL_NAME not found!"
src19 = src19.replace(OLD_MODEL_NAME, NEW_MODEL_NAME)

set_cell_src(cell19_idx, src19)
c19 = cell_src(cell19_idx)
assert "'num_leaves': 255" in c19
assert 'leaves255' in c19
print("Cell 19: OK")

for cell in nb['cells']:
    if 'id' not in cell:
        cell['id'] = str(uuid.uuid4())[:8]
for cell in nb['cells']:
    if cell.get('cell_type') == 'code':
        cell['outputs'] = []
        cell['execution_count'] = None

with open(DST_NB, 'w') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print(f"\nv05d34 notebook written: {DST_NB}")
