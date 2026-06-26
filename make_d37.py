"""Create v05d37 from v05d35 (new best):
winner_weight 4x → 6x, keeping phase_weight scaling.

d35 was the new best (0.0582 vs d25 0.0555) with:
  winner_wt=4x × phase_weight[0.2, 0.5, 1.0]

d37 pushes winner_weight to 6x:
  winner+mc=1.0: 6.0 (up from 4.0)
  winner+mc=0.5: 3.0 (up from 2.0)
  winner+mc=0.2: 1.2 (up from 0.8)
  loser +mc=1.0: 1.0 (unchanged)
  loser +mc=0.2: 0.2 (unchanged)

d27 tried 8x without phase_weight and got 0.0491 (worse than d25's 4x).
But phase_weight scaling changes the effective weight distribution, so
6x×phase_wt may behave differently from raw 6x.
"""
import json, uuid
from pathlib import Path

WORK_DIR = Path('/kaggle/working')
SRC_NB = WORK_DIR / 'pokemon-20260625-v0-05d35.ipynb'  # build on d35 (new best)
DST_NB = WORK_DIR / 'pokemon-20260625-v0-05d37.ipynb'

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
src1 = src1.replace("EXPERIMENT_NAME = 'v0_05d35_lgbm_mcwt_correct'",
                    "EXPERIMENT_NAME = 'v0_05d37_lgbm_wt6_phwt'")
src1 = src1.replace("RUN_PREFIX = 'pokemon-20260625-v0-05d35'",
                    "RUN_PREFIX = 'pokemon-20260625-v0-05d37'")
set_cell_src(1, src1)
assert 'v0_05d37_lgbm_wt6_phwt' in cell_src(1)
print("Cell 1: OK")

cell19_idx = None
for i, cell in enumerate(nb['cells']):
    src = ''.join(cell['source'])
    if 'lambdarank' in src and 'UNKNOWN0_MLP_REPORT' in src:
        cell19_idx = i
        break
assert cell19_idx is not None
src19 = cell_src(cell19_idx)

# ── Patch: winner_weight 4.0 → 6.0 ──────────────────────────────────────────
assert "_winner_weight = 4.0" in src19
src19 = src19.replace(
    "                _winner_weight = 4.0\n"
    "                train_df['_win_wt'] = (train_df['won'].fillna(0).astype(float) * (_winner_weight - 1.0) + 1.0\n"
    "                                       ) * train_df['_phase_wt'].clip(lower=0.1)",
    "                _winner_weight = 6.0\n"
    "                train_df['_win_wt'] = (train_df['won'].fillna(0).astype(float) * (_winner_weight - 1.0) + 1.0\n"
    "                                       ) * train_df['_phase_wt'].clip(lower=0.1)"
)
assert "_winner_weight = 6.0" in src19, "winner_weight patch failed!"

OLD_MODEL_NAME = "                    'model': 'lightgbm_lambdarank_mcwt_correct',"
NEW_MODEL_NAME = "                    'model': 'lightgbm_lambdarank_wt6_phwt',"
assert OLD_MODEL_NAME in src19
src19 = src19.replace(OLD_MODEL_NAME, NEW_MODEL_NAME)

set_cell_src(cell19_idx, src19)
c19 = cell_src(cell19_idx)
assert '_winner_weight = 6.0' in c19 and 'wt6_phwt' in c19
print("Cell 19: OK (winner_weight=6.0)")

for cell in nb['cells']:
    if 'id' not in cell:
        cell['id'] = str(uuid.uuid4())[:8]
for cell in nb['cells']:
    if cell.get('cell_type') == 'code':
        cell['outputs'] = []
        cell['execution_count'] = None

with open(DST_NB, 'w') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print(f"\nv05d37 notebook written: {DST_NB}")
