"""Create v05d38 from v05d36 (new best: 0.0601):
Combine d35 (phase_weight × winner_wt) + d36 (feature engineering).

d36 (features only): 0.0601  +0.0046 vs d25
d35 (phase_wt only): 0.0582  +0.0027 vs d25

Both improve independently. Combination should stack:
  winner_wt=4x × phase_weight[0.2, 0.5, 1.0] + can_attack/hp_diff/energy_surplus/hand_advantage/bench_lead
"""
import json, uuid
from pathlib import Path

WORK_DIR = Path('/kaggle/working')
SRC_NB = WORK_DIR / 'pokemon-20260625-v0-05d36.ipynb'  # base: d36 (has feature eng)
DST_NB = WORK_DIR / 'pokemon-20260625-v0-05d38.ipynb'

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
                    "EXPERIMENT_NAME = 'v0_05d38_lgbm_phwt_feateng'")
src1 = src1.replace("RUN_PREFIX = 'pokemon-20260625-v0-05d36'",
                    "RUN_PREFIX = 'pokemon-20260625-v0-05d38'")
set_cell_src(1, src1)
assert 'v0_05d38_lgbm_phwt_feateng' in cell_src(1)
print("Cell 1: OK")

cell19_idx = None
for i, cell in enumerate(nb['cells']):
    src = ''.join(cell['source'])
    if 'lambdarank' in src and 'UNKNOWN0_MLP_REPORT' in src:
        cell19_idx = i
        break
assert cell19_idx is not None
src19 = cell_src(cell19_idx)

# d36 already has feature engineering and plain winner_wt=4x
# We need to ADD phase_weight merging (from d35) to the winner_wt calculation
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
    "            # Winner-weighted * phase_weight (d35 + d36 combo)\n"
    "            train_df = train_df.copy()\n"
    "            if 'ALAKAZAM_OPTION_MODEL_DF' in globals() and 'phase_weight' in ALAKAZAM_OPTION_MODEL_DF.columns:\n"
    "                _u0_adf = ALAKAZAM_OPTION_MODEL_DF[ALAKAZAM_OPTION_MODEL_DF['context_name']=='UNKNOWN_0']\n"
    "                _dec_pw_map = _u0_adf.groupby('decision_id')['phase_weight'].first()\n"
    "                train_df['_phase_wt'] = train_df['decision_id'].map(_dec_pw_map).fillna(0.5)\n"
    "                _pw_vals = train_df['_phase_wt'].value_counts().head(5).to_dict()\n"
    "                print(f'Phase weight merged: {_pw_vals}')\n"
    "            else:\n"
    "                train_df['_phase_wt'] = 0.5\n"
    "                print('phase_weight not available, using 0.5')\n"
    "            if 'won' in train_df.columns:\n"
    "                _winner_weight = 4.0\n"
    "                train_df['_win_wt'] = (train_df['won'].fillna(0).astype(float) * (_winner_weight - 1.0) + 1.0\n"
    "                                       ) * train_df['_phase_wt'].clip(lower=0.1)\n"
    "                _n_winner = int((train_df['won'] == True).sum())\n"
    "                _pw_mean = float(train_df['_phase_wt'].mean())\n"
    "                print(f'Winner*phase_wt: {_n_winner}/{len(train_df)} winner rows, pw_mean={_pw_mean:.3f}')\n"
    "            else:\n"
    "                train_df['_win_wt'] = train_df['_phase_wt'].clip(lower=0.1)\n"
    "                print('won column not found, using phase_weight as weight')"
)
assert OLD_WT in src19, "OLD_WT not found!"
src19 = src19.replace(OLD_WT, NEW_WT)

OLD_MODEL_NAME = "                    'model': 'lightgbm_lambdarank_feateng',"
NEW_MODEL_NAME = "                    'model': 'lightgbm_lambdarank_phwt_feateng',"
assert OLD_MODEL_NAME in src19
src19 = src19.replace(OLD_MODEL_NAME, NEW_MODEL_NAME)

set_cell_src(cell19_idx, src19)
c19 = cell_src(cell19_idx)
assert '_phase_wt' in c19 and 'phwt_feateng' in c19 and 'can_attack' in c19
print("Cell 19: OK (phase_wt + feature engineering)")

for cell in nb['cells']:
    if 'id' not in cell:
        cell['id'] = str(uuid.uuid4())[:8]
for cell in nb['cells']:
    if cell.get('cell_type') == 'code':
        cell['outputs'] = []
        cell['execution_count'] = None

with open(DST_NB, 'w') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print(f"\nv05d38 notebook written: {DST_NB}")
