"""Create v05d35 from v05d25:
mc_step_weight training weight — CORRECT version (uses Cell-13 decision-level values).

d32 FAILED because it used the pre-Cell-13 mc_step_weight (gamma-discount, [0, 0.65])
from UNKNOWN0_TRAIN_VALID_DF. This is ~zero for most rows.

Fix: in Cell 19, merge the CORRECT decision-level mc_step_weight from
ALAKAZAM_OPTION_MODEL_DF (which has Cell-13 delta-V values [0.2, 0.5, 1.0]).

After merge:
  winner, mc=1.0  → _win_wt = 4.0 * 1.0 = 4.0  (critical end-of-turn decisions)
  winner, mc=0.5  → _win_wt = 4.0 * 0.5 = 2.0
  winner, mc=0.2  → _win_wt = 4.0 * 0.2 = 0.8  (trivial decisions, soft-ignored)
  loser,  mc=1.0  → _win_wt = 1.0 * 1.0 = 1.0
  loser,  mc=0.2  → _win_wt = 1.0 * 0.2 = 0.2  (near-ignored)
"""
import json, uuid
from pathlib import Path

WORK_DIR = Path('/kaggle/working')
SRC_NB = WORK_DIR / 'pokemon-20260625-v0-05d25.ipynb'
DST_NB = WORK_DIR / 'pokemon-20260625-v0-05d35.ipynb'

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
                    "EXPERIMENT_NAME = 'v0_05d35_lgbm_mcwt_correct'")
src1 = src1.replace("RUN_PREFIX = 'pokemon-20260625-v0-05d25'",
                    "RUN_PREFIX = 'pokemon-20260625-v0-05d35'")
set_cell_src(1, src1)
assert 'v0_05d35_lgbm_mcwt_correct' in cell_src(1)
print("Cell 1: OK")

cell19_idx = None
for i, cell in enumerate(nb['cells']):
    src = ''.join(cell['source'])
    if 'lambdarank' in src and 'UNKNOWN0_MLP_REPORT' in src:
        cell19_idx = i
        break
assert cell19_idx is not None
src19 = cell_src(cell19_idx)

# ── Patch: winner_weight * CORRECT decision-level mc_step_weight ──────────────
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
    "            # Winner-weighted * decision-level mc_step_weight (from Cell-13 delta-V values)\n"
    "            train_df = train_df.copy()\n"
    "            # Merge correct decision-level mc_step_weight [0.2, 0.5, 1.0] from ALAKAZAM_OPTION_MODEL_DF\n"
    "            if 'ALAKAZAM_OPTION_MODEL_DF' in globals() and 'mc_step_weight' in ALAKAZAM_OPTION_MODEL_DF.columns:\n"
    "                _u0_adf = ALAKAZAM_OPTION_MODEL_DF[ALAKAZAM_OPTION_MODEL_DF['context_name']=='UNKNOWN_0']\n"
    "                _dec_mc_map = _u0_adf.groupby('decision_id')['mc_step_weight'].first()\n"
    "                train_df['_dec_mc_sw'] = train_df['decision_id'].map(_dec_mc_map).fillna(0.5)\n"
    "                _mc_vals = train_df['_dec_mc_sw'].value_counts().head(5).to_dict()\n"
    "                print(f'Decision-level mc_step_weight merged: {_mc_vals}')\n"
    "            else:\n"
    "                train_df['_dec_mc_sw'] = 0.5\n"
    "                print('ALAKAZAM_OPTION_MODEL_DF not available, using uniform mc_step_weight=0.5')\n"
    "            if 'won' in train_df.columns:\n"
    "                _winner_weight = 4.0\n"
    "                train_df['_win_wt'] = (train_df['won'].fillna(0).astype(float) * (_winner_weight - 1.0) + 1.0\n"
    "                                       ) * train_df['_dec_mc_sw'].clip(lower=0.1)\n"
    "                _n_winner = int((train_df['won'] == True).sum())\n"
    "                _mc_mean = float(train_df['_dec_mc_sw'].mean())\n"
    "                print(f'Winner*mc_step: {_n_winner}/{len(train_df)} winner rows, dec_mc_mean={_mc_mean:.3f}')\n"
    "            else:\n"
    "                train_df['_win_wt'] = train_df['_dec_mc_sw'].clip(lower=0.1)\n"
    "                print('won column not found, using mc_step_weight as weight')"
)
assert OLD_WT in src19, "OLD_WT not found!"
src19 = src19.replace(OLD_WT, NEW_WT)

OLD_MODEL_NAME = "                    'model': 'lightgbm_lambdarank_winner_wt4',"
NEW_MODEL_NAME = "                    'model': 'lightgbm_lambdarank_mcwt_correct',"
assert OLD_MODEL_NAME in src19
src19 = src19.replace(OLD_MODEL_NAME, NEW_MODEL_NAME)

set_cell_src(cell19_idx, src19)
c19 = cell_src(cell19_idx)
assert '_dec_mc_sw' in c19 and 'mcwt_correct' in c19
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
print(f"\nv05d35 notebook written: {DST_NB}")
