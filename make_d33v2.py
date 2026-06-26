"""Create v05d33 (v2) from v05d25:
Filter training to mc_step_weight >= 0.5 — CORRECT version.

d33 original was broken: used pre-Cell-13 mc_step_weight (gamma-discount, [0, 0.65]).
This version uses the CORRECT decision-level mc_step_weight [0.2, 0.5, 1.0] from
ALAKAZAM_OPTION_MODEL_DF (post-Cell-13 delta-V values).

Filter: remove decisions where delta-V importance < 0.5 (the trivial mc=0.2 decisions).
This keeps 55.1% of training decisions (mc=0.5 and mc=1.0 only).
"""
import json, uuid
from pathlib import Path

WORK_DIR = Path('/kaggle/working')
SRC_NB = WORK_DIR / 'pokemon-20260625-v0-05d25.ipynb'
DST_NB = WORK_DIR / 'pokemon-20260625-v0-05d33.ipynb'  # overwrite broken d33

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

# ── Patch: filter to decision-level mc_step_weight >= 0.5 ────────────────────
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
    "            # Filter to mc_step_weight >= 0.5 using CORRECT decision-level values from adf\n"
    "            train_df = train_df.copy()\n"
    "            if 'ALAKAZAM_OPTION_MODEL_DF' in globals() and 'mc_step_weight' in ALAKAZAM_OPTION_MODEL_DF.columns:\n"
    "                _u0_adf = ALAKAZAM_OPTION_MODEL_DF[ALAKAZAM_OPTION_MODEL_DF['context_name']=='UNKNOWN_0']\n"
    "                _dec_mc_map = _u0_adf.groupby('decision_id')['mc_step_weight'].first()\n"
    "                _n_before = len(train_df)\n"
    "                _dec_mc_ser = train_df['decision_id'].map(_dec_mc_map).fillna(0.5)\n"
    "                train_df = train_df[_dec_mc_ser >= 0.5].copy()\n"
    "                print(f'mc_step_weight>=0.5 filter: {len(train_df)}/{_n_before} rows kept '\n"
    "                      f'({100*len(train_df)/_n_before:.1f}%)')\n"
    "            else:\n"
    "                print('ALAKAZAM_OPTION_MODEL_DF not available, skipping mc_step_weight filter')\n"
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

OLD_MODEL_NAME = "                    'model': 'lightgbm_lambdarank_winner_wt4',"
NEW_MODEL_NAME = "                    'model': 'lightgbm_lambdarank_mcwt_filter05',"
assert OLD_MODEL_NAME in src19
src19 = src19.replace(OLD_MODEL_NAME, NEW_MODEL_NAME)

set_cell_src(cell19_idx, src19)
c19 = cell_src(cell19_idx)
assert '_dec_mc_map' in c19 and 'mcwt_filter05' in c19
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
print(f"\nv05d33 (v2) notebook written: {DST_NB}")
