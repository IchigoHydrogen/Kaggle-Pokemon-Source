"""Create v05d33 from v05d25:
Filter training to mc_step_weight >= 0.5 (remove trivial decisions).

mc_step_weight distribution: 0.2=44.9% (trivial), 0.5=16.6% (medium), 1.0=38.4% (critical)
Current training mixes trivial decisions (mc_step=0.2) with critical ones.
Trivial decisions: often forced/auto moves mid-turn → noisy labels from winner/loser.

After filter: 55.1% of data remains (only medium + critical decisions).
  ~243k / 441k rows → training on decisions that actually determine the game.

Hypothesis: removing 44.9% noise rows improves signal quality enough to outweigh data loss.
Aggressive version: if this works, d34 will try mc_step_weight==1.0 only (38.4% of data).
"""
import json, uuid
from pathlib import Path

WORK_DIR = Path('/kaggle/working')
SRC_NB = WORK_DIR / 'pokemon-20260625-v0-05d25.ipynb'
DST_NB = WORK_DIR / 'pokemon-20260625-v0-05d33.ipynb'

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
                    "EXPERIMENT_NAME = 'v0_05d33_lgbm_mcwt_filter05'")
src1 = src1.replace("RUN_PREFIX = 'pokemon-20260625-v0-05d25'",
                    "RUN_PREFIX = 'pokemon-20260625-v0-05d33'")
set_cell_src(1, src1)
assert 'v0_05d33_lgbm_mcwt_filter05' in cell_src(1)
print("Cell 1: OK")

cell19_idx = None
for i, cell in enumerate(nb['cells']):
    src = ''.join(cell['source'])
    if 'lambdarank' in src and 'UNKNOWN0_MLP_REPORT' in src:
        cell19_idx = i
        break
assert cell19_idx is not None
src19 = cell_src(cell19_idx)

# ── Patch 1: filter train_df to mc_step_weight >= 0.5 before weight calc ─────
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
    "            # Filter to mc_step_weight >= 0.5: remove trivial auto-decisions\n"
    "            train_df = train_df.copy()\n"
    "            if 'mc_step_weight' in train_df.columns:\n"
    "                _n_before = len(train_df)\n"
    "                # Group by decision_id so whole decision is kept/dropped atomically\n"
    "                _dec_mc = train_df.groupby('decision_id')['mc_step_weight'].first()\n"
    "                _keep_dec = set(_dec_mc[_dec_mc >= 0.5].index)\n"
    "                train_df = train_df[train_df['decision_id'].isin(_keep_dec)].copy()\n"
    "                print(f'mc_step_weight>=0.5 filter: {len(train_df)}/{_n_before} rows kept '\n"
    "                      f'({100*len(train_df)/_n_before:.1f}%)')\n"
    "            if 'won' in train_df.columns:\n"
    "                _winner_weight = 4.0\n"
    "                train_df['_win_wt'] = train_df['won'].fillna(0).astype(float) * (_winner_weight - 1.0) + 1.0\n"
    "                _n_winner = int((train_df['won'] == True).sum())\n"
    "                print(f'Winner-weighted training: {_n_winner}/{len(train_df)} winner rows (weight={_winner_weight}x)')\n"
    "            else:\n"
    "                train_df['_win_wt'] = 1.0\n"
    "                print('won column not found, using uniform weights')"
)
assert OLD_WT in src19, "OLD_WT not found!"
src19 = src19.replace(OLD_WT, NEW_WT)

# ── Patch 2: model name ───────────────────────────────────────────────────────
OLD_MODEL_NAME = "                    'model': 'lightgbm_lambdarank_winner_wt4',"
NEW_MODEL_NAME = "                    'model': 'lightgbm_lambdarank_mcwt_filter05',"
assert OLD_MODEL_NAME in src19, "OLD_MODEL_NAME not found!"
src19 = src19.replace(OLD_MODEL_NAME, NEW_MODEL_NAME)

set_cell_src(cell19_idx, src19)
c19 = cell_src(cell19_idx)
assert 'mc_step_weight>=0.5' in c19 or 'mc_step_weight' in c19
assert 'mcwt_filter05' in c19
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
print(f"\nv05d33 notebook written: {DST_NB}")
