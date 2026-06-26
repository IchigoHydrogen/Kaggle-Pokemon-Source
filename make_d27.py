"""Create v05d27 from v05d25:
Push winner weight 4x → 8x. d25 (4x, best_iter=161) showed the trend continues.
Also widen early_stopping 50 → 100 to let the model train longer.
Revert to T200_T200 filter (d26 showed T200_OTHER pollutes training).
"""
import json, uuid
from pathlib import Path

WORK_DIR = Path('/kaggle/working')
SRC_NB = WORK_DIR / 'pokemon-20260625-v0-05d25.ipynb'
DST_NB = WORK_DIR / 'pokemon-20260625-v0-05d27.ipynb'

with open(SRC_NB) as f:
    nb = json.load(f)

def cell_src(i):
    return ''.join(nb['cells'][i]['source'])

def set_cell_src(i, s):
    nb['cells'][i]['source'] = s
    nb['cells'][i]['outputs'] = []
    if nb['cells'][i].get('cell_type') == 'code':
        nb['cells'][i]['execution_count'] = None

# ── Cell 1: version bump (TOP200_EPISODE_FILTER stays True via default) ───────
src1 = cell_src(1)
src1 = src1.replace("EXPERIMENT_NAME = 'v0_05d25_lgbm_winner_wt4'",
                    "EXPERIMENT_NAME = 'v0_05d27_lgbm_winner_wt8'")
src1 = src1.replace("RUN_PREFIX = 'pokemon-20260625-v0-05d25'",
                    "RUN_PREFIX = 'pokemon-20260625-v0-05d27'")
set_cell_src(1, src1)
assert 'v0_05d27_lgbm_winner_wt8' in cell_src(1)
assert 'pokemon-20260625-v0-05d27' in cell_src(1)
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

# ── Patch 1: winner_weight 4.0 → 8.0 ─────────────────────────────────────────
OLD_WT = "                _winner_weight = 4.0"
NEW_WT = "                _winner_weight = 8.0"
assert OLD_WT in src19, "OLD_WT not found!"
src19 = src19.replace(OLD_WT, NEW_WT)

# ── Patch 2: early_stopping 50 → 100 ─────────────────────────────────────────
OLD_ES = "                        _lgb_mod.early_stopping(50, verbose=False),"
NEW_ES = "                        _lgb_mod.early_stopping(100, verbose=False),"
assert OLD_ES in src19, "OLD_ES not found!"
src19 = src19.replace(OLD_ES, NEW_ES)

# ── Patch 3: model name ───────────────────────────────────────────────────────
OLD_MODEL_NAME = "                    'model': 'lightgbm_lambdarank_winner_wt4',"
NEW_MODEL_NAME = "                    'model': 'lightgbm_lambdarank_winner_wt8',"
assert OLD_MODEL_NAME in src19, "OLD_MODEL_NAME not found!"
src19 = src19.replace(OLD_MODEL_NAME, NEW_MODEL_NAME)

set_cell_src(cell19_idx, src19)

c19 = cell_src(cell19_idx)
assert '_winner_weight = 8.0' in c19
assert 'early_stopping(100' in c19
assert 'winner_wt8' in c19
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
print(f"\nv05d27 notebook written: {DST_NB}")
