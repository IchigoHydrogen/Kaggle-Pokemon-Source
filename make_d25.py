"""Create v05d25 from v05d24:
Push winner weighting from 2x → 4x. d22 (winner-only=∞x) gave 0.0494,
d24 (2x) gave 0.0515. Need to find if 4x is better than 2x.

Also increase early_stopping_rounds 30→50 to allow more training with higher weighting.
"""
import json, uuid
from pathlib import Path

WORK_DIR = Path('/kaggle/working')
SRC_NB = WORK_DIR / 'pokemon-20260625-v0-05d24.ipynb'
DST_NB = WORK_DIR / 'pokemon-20260625-v0-05d25.ipynb'

with open(SRC_NB) as f:
    nb = json.load(f)

def cell_src(i):
    return ''.join(nb['cells'][i]['source'])

def set_cell_src(i, s):
    nb['cells'][i]['source'] = s
    nb['cells'][i]['outputs'] = []
    if nb['cells'][i].get('cell_type') == 'code':
        nb['cells'][i]['execution_count'] = None

# ── Cell 1: version bump ─────────────────────────────────────────────────────
src1 = cell_src(1)
src1 = src1.replace("EXPERIMENT_NAME = 'v0_05d24_lgbm_winner_wt2'",
                    "EXPERIMENT_NAME = 'v0_05d25_lgbm_winner_wt4'")
src1 = src1.replace("RUN_PREFIX = 'pokemon-20260625-v0-05d24'",
                    "RUN_PREFIX = 'pokemon-20260625-v0-05d25'")
set_cell_src(1, src1)
assert 'v0_05d25_lgbm_winner_wt4' in cell_src(1)
assert 'pokemon-20260625-v0-05d25' in cell_src(1)
print("Cell 1: OK")

# ── Find Cell 19 ─────────────────────────────────────────────────────────────
cell19_idx = None
for i, cell in enumerate(nb['cells']):
    src = ''.join(cell['source'])
    if 'lambdarank' in src and 'UNKNOWN0_MLP_REPORT' in src:
        cell19_idx = i
        break
if cell19_idx is None:
    raise RuntimeError("Could not find Cell 19")
print(f"Cell 19 found at index {cell19_idx}")

src19 = cell_src(cell19_idx)

# ── Patch 1: bump winner_weight 2.0 → 4.0 ────────────────────────────────────
OLD_WINNER_WT = "                _winner_weight = 2.0"
NEW_WINNER_WT = "                _winner_weight = 4.0"
assert OLD_WINNER_WT in src19, "OLD_WINNER_WT not found!"
src19 = src19.replace(OLD_WINNER_WT, NEW_WINNER_WT)

# ── Patch 2: increase early stopping rounds 30 → 50 ──────────────────────────
OLD_ES = "                        _lgb_mod.early_stopping(30, verbose=False),"
NEW_ES = "                        _lgb_mod.early_stopping(50, verbose=False),"
assert OLD_ES in src19, "OLD_ES not found!"
src19 = src19.replace(OLD_ES, NEW_ES)

# ── Patch 3: update model name ─────────────────────────────────────────────────
OLD_MODEL_NAME = "                    'model': 'lightgbm_lambdarank_winner_wt2',"
NEW_MODEL_NAME = "                    'model': 'lightgbm_lambdarank_winner_wt4',"
assert OLD_MODEL_NAME in src19, "OLD_MODEL_NAME not found!"
src19 = src19.replace(OLD_MODEL_NAME, NEW_MODEL_NAME)

set_cell_src(cell19_idx, src19)

c19 = cell_src(cell19_idx)
assert '_winner_weight = 4.0' in c19, "winner weight not updated!"
assert 'early_stopping(50' in c19, "early stopping not updated!"
assert 'winner_wt4' in c19, "model name not updated!"
print("Cell 19: OK")

# ── Add IDs / clear outputs ───────────────────────────────────────────────────
for cell in nb['cells']:
    if 'id' not in cell:
        cell['id'] = str(uuid.uuid4())[:8]
for cell in nb['cells']:
    if cell.get('cell_type') == 'code':
        cell['outputs'] = []
        cell['execution_count'] = None

with open(DST_NB, 'w') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print(f"\nv05d25 notebook written: {DST_NB}")
print(f"Total cells: {len(nb['cells'])}")
