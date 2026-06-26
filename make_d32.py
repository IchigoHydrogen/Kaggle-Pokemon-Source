"""Create v05d32 from v05d25:
mc_step_weight as training weight modifier.

Addresses 'decision importance not being distinguished':
  winner at mc_step_weight=1.0 → weight = 4.0 * 1.0 = 4.0  (max, critical end-of-turn)
  winner at mc_step_weight=0.5 → weight = 4.0 * 0.5 = 2.0
  winner at mc_step_weight=0.2 → weight = 4.0 * 0.2 = 0.8  (trivial, essentially ignored)
  loser  at mc_step_weight=1.0 → weight = 1.0 * 1.0 = 1.0
  loser  at mc_step_weight=0.2 → weight = 1.0 * 0.2 = 0.2  (effectively ignored)

mc_step_weight distribution in training: 0.2=44.9%, 0.5=16.6%, 1.0=38.4%
(bimodal: trivial decisions vs critical end-of-turn decisions)

No sig routing, no eval override, no feature changes. Pure weight change.
"""
import json, uuid
from pathlib import Path

WORK_DIR = Path('/kaggle/working')
SRC_NB = WORK_DIR / 'pokemon-20260625-v0-05d25.ipynb'
DST_NB = WORK_DIR / 'pokemon-20260625-v0-05d32.ipynb'

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
                    "EXPERIMENT_NAME = 'v0_05d32_lgbm_mcwt_weight'")
src1 = src1.replace("RUN_PREFIX = 'pokemon-20260625-v0-05d25'",
                    "RUN_PREFIX = 'pokemon-20260625-v0-05d32'")
set_cell_src(1, src1)
assert 'v0_05d32_lgbm_mcwt_weight' in cell_src(1)
print("Cell 1: OK")

cell19_idx = None
for i, cell in enumerate(nb['cells']):
    src = ''.join(cell['source'])
    if 'lambdarank' in src and 'UNKNOWN0_MLP_REPORT' in src:
        cell19_idx = i
        break
assert cell19_idx is not None
print(f"Cell 19 found at index {cell19_idx}")
src19 = cell_src(cell19_idx)

# ── Patch 1: multiply _win_wt by mc_step_weight ───────────────────────────────
OLD_WT = (
    "            # Winner-weighted training: winner decisions get 2x weight\n"
    "            if 'won' in train_df.columns:\n"
    "                _winner_weight = 4.0\n"
    "                train_df = train_df.copy()\n"
    "                train_df['_win_wt'] = train_df['won'].fillna(0).astype(float) * (_winner_weight - 1.0) + 1.0\n"
    "                _n_winner = int((train_df['won'] == True).sum())\n"
    "                print(f'Winner-weighted training: {_n_winner}/{len(train_df)} winner rows (weight={_winner_weight}x)')\n"
    "            else:\n"
    "                train_df = train_df.copy()\n"
    "                train_df['_win_wt'] = 1.0\n"
    "                print('won column not found, using uniform weights')"
)
NEW_WT = (
    "            # Winner-weighted * mc_step_weight: critical decisions emphasized\n"
    "            train_df = train_df.copy()\n"
    "            if 'won' in train_df.columns:\n"
    "                _winner_weight = 4.0\n"
    "                train_df['_win_wt'] = train_df['won'].fillna(0).astype(float) * (_winner_weight - 1.0) + 1.0\n"
    "                # Scale by decision importance: trivial (0.2) shrinks to 0.2x, critical (1.0) unchanged\n"
    "                if 'mc_step_weight' in train_df.columns:\n"
    "                    _mc = train_df['mc_step_weight'].fillna(0.2).clip(lower=0.1)\n"
    "                    train_df['_win_wt'] = train_df['_win_wt'] * _mc\n"
    "                    _n_winner = int((train_df['won'] == True).sum())\n"
    "                    _mc_mean = float(_mc.mean())\n"
    "                    print(f'mc_step_weight weight: {_n_winner}/{len(train_df)} winner rows, mc_mean={_mc_mean:.3f}')\n"
    "                else:\n"
    "                    _n_winner = int((train_df['won'] == True).sum())\n"
    "                    print(f'Winner-weighted: {_n_winner}/{len(train_df)} winner rows (mc_step_weight not found)')\n"
    "            else:\n"
    "                train_df['_win_wt'] = 1.0\n"
    "                print('won column not found, using uniform weights')"
)
assert OLD_WT in src19, "OLD_WT not found!"
src19 = src19.replace(OLD_WT, NEW_WT)

# ── Patch 2: model name ───────────────────────────────────────────────────────
OLD_MODEL_NAME = "                    'model': 'lightgbm_lambdarank_winner_wt4',"
NEW_MODEL_NAME = "                    'model': 'lightgbm_lambdarank_mcwt_weight',"
assert OLD_MODEL_NAME in src19, "OLD_MODEL_NAME not found!"
src19 = src19.replace(OLD_MODEL_NAME, NEW_MODEL_NAME)

set_cell_src(cell19_idx, src19)

c19 = cell_src(cell19_idx)
assert 'mc_step_weight' in c19 and 'mc_mean' in c19
assert 'mcwt_weight' in c19
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
print(f"\nv05d32 notebook written: {DST_NB}")
