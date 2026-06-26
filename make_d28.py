"""Create v05d28 from v05d25:
Expand training budget: max_rows 350k → 500k.

T200_T200 UNKNOWN_0 data has 441k rows. Current cap of 350k forces sampling
to 79% of available data. Increasing to 500k uses ALL 441k rows → ~374k
training rows (vs current 297k, +26%).

Keep winner_weight=4x (confirmed best in d25), early_stopping=50.
"""
import json, uuid
from pathlib import Path

WORK_DIR = Path('/kaggle/working')
SRC_NB = WORK_DIR / 'pokemon-20260625-v0-05d25.ipynb'
DST_NB = WORK_DIR / 'pokemon-20260625-v0-05d28.ipynb'

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
src1 = src1.replace("EXPERIMENT_NAME = 'v0_05d25_lgbm_winner_wt4'",
                    "EXPERIMENT_NAME = 'v0_05d28_lgbm_maxrows500k'")
src1 = src1.replace("RUN_PREFIX = 'pokemon-20260625-v0-05d25'",
                    "RUN_PREFIX = 'pokemon-20260625-v0-05d28'")
set_cell_src(1, src1)
assert 'v0_05d28_lgbm_maxrows500k' in cell_src(1)
assert 'pokemon-20260625-v0-05d28' in cell_src(1)
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

# ── Patch: expand max_rows cap 350_000 → 500_000, use MLP_MAX_ROWS directly ──
OLD_MAXROWS = (
    "        max_rows = int(os.environ.get('V05_UNKNOWN0_MLP_MAX_ROWS',\n"
    "                       str(min(350_000, max(50_000, MLP_MAX_ROWS // 2)))))"
)
NEW_MAXROWS = (
    "        max_rows = int(os.environ.get('V05_UNKNOWN0_MLP_MAX_ROWS',\n"
    "                       str(min(500_000, max(50_000, MLP_MAX_ROWS)))))"
)
assert OLD_MAXROWS in src19, f"OLD_MAXROWS not found!\nSearching for:\n{OLD_MAXROWS}"
src19 = src19.replace(OLD_MAXROWS, NEW_MAXROWS)

# ── Update model name ─────────────────────────────────────────────────────────
OLD_MODEL_NAME = "                    'model': 'lightgbm_lambdarank_winner_wt4',"
NEW_MODEL_NAME = "                    'model': 'lightgbm_lambdarank_maxrows500k',"
assert OLD_MODEL_NAME in src19, "OLD_MODEL_NAME not found!"
src19 = src19.replace(OLD_MODEL_NAME, NEW_MODEL_NAME)

set_cell_src(cell19_idx, src19)

c19 = cell_src(cell19_idx)
assert '500_000' in c19, "max_rows not updated!"
assert 'maxrows500k' in c19, "model name not updated!"
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
print(f"\nv05d28 notebook written: {DST_NB}")
