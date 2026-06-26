"""Create v05d39 from v05d36 (best: 0.0601):
Add more computed features on top of d36's 5 features:

  hp_ratio         = my_active_hp / (my_active_hp + op_active_hp + 1)
  turn_normalized  = min(turn / 30.0, 1.0)
  energy_ratio     = my_active_energy_count / max(remain_energy_cost, 1)
  deck_ratio       = my_deck_count / (my_deck_count + op_deck_count + 1)
  total_poke_lead  = (my_bench_count + 1) - (op_bench_count + 1)  = bench_lead (diff from bench only)

All computable at inference time from existing obs fields.
"""
import json, uuid
from pathlib import Path

WORK_DIR = Path('/kaggle/working')
SRC_NB = WORK_DIR / 'pokemon-20260625-v0-05d36.ipynb'  # base: d36 (best)
DST_NB = WORK_DIR / 'pokemon-20260625-v0-05d39.ipynb'

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
                    "EXPERIMENT_NAME = 'v0_05d39_lgbm_feateng2'")
src1 = src1.replace("RUN_PREFIX = 'pokemon-20260625-v0-05d36'",
                    "RUN_PREFIX = 'pokemon-20260625-v0-05d39'")
set_cell_src(1, src1)
assert 'v0_05d39_lgbm_feateng2' in cell_src(1)
print("Cell 1: OK")

cell19_idx = None
for i, cell in enumerate(nb['cells']):
    src = ''.join(cell['source'])
    if 'lambdarank' in src and 'UNKNOWN0_MLP_REPORT' in src:
        cell19_idx = i
        break
assert cell19_idx is not None
src19 = cell_src(cell19_idx)

# ── Patch 1: extend UNKNOWN0_NUMERIC_FEATURES with 4 more features ────────────
OLD_FEAT = (
    "    # v05d36: interaction features\n"
    "    'can_attack', 'hp_diff', 'energy_surplus', 'hand_advantage', 'bench_lead',\n"
    "]"
)
NEW_FEAT = (
    "    # v05d36: interaction features\n"
    "    'can_attack', 'hp_diff', 'energy_surplus', 'hand_advantage', 'bench_lead',\n"
    "    # v05d39: ratio/normalized features\n"
    "    'hp_ratio', 'turn_normalized', 'energy_ratio', 'deck_ratio',\n"
    "]"
)
assert OLD_FEAT in src19, "OLD_FEAT not found!"
src19 = src19.replace(OLD_FEAT, NEW_FEAT)

# ── Patch 2: compute new features in train_df/valid_df ────────────────────────
OLD_PREP = (
    "            print('Feature engineering: can_attack, hp_diff, energy_surplus, hand_advantage, bench_lead')\n"
)
NEW_PREP = (
    "            print('Feature engineering: can_attack, hp_diff, energy_surplus, hand_advantage, bench_lead')\n"
    "                _fdf['hp_ratio'] = _fdf['my_active_hp'].fillna(0) / (_fdf['my_active_hp'].fillna(0) + _fdf['op_active_hp'].fillna(0) + 1.0)\n"
    "                _fdf['turn_normalized'] = (_fdf['turn'].fillna(0) / 30.0).clip(upper=1.0)\n"
    "                _fdf['energy_ratio'] = _fdf['my_active_energy_count'].fillna(0) / (_fdf['remain_energy_cost'].fillna(1).clip(lower=1))\n"
    "                _fdf['deck_ratio'] = _fdf['my_deck_count'].fillna(0) / (_fdf['my_deck_count'].fillna(0) + _fdf['op_deck_count'].fillna(0) + 1.0)\n"
    "            print('Feature engineering v2: hp_ratio, turn_normalized, energy_ratio, deck_ratio')\n"
)
assert OLD_PREP in src19, "OLD_PREP not found!"
src19 = src19.replace(OLD_PREP, NEW_PREP)

# ── Patch 3: model name ───────────────────────────────────────────────────────
OLD_MODEL_NAME = "                    'model': 'lightgbm_lambdarank_feateng',"
NEW_MODEL_NAME = "                    'model': 'lightgbm_lambdarank_feateng2',"
assert OLD_MODEL_NAME in src19
src19 = src19.replace(OLD_MODEL_NAME, NEW_MODEL_NAME)

# ── Patch 4: update _LGBM_INJ_CODE for new features ─────────────────────────
OLD_INJ = (
    "        # v05d36: add interaction features\\n"
    "        for _r in rows:\\n"
    "            _r[\\'can_attack\\'] = float(_r[\\'my_active_energy_count\\'] >= _r[\\'remain_energy_cost\\'])\\n"
    "            _r[\\'hp_diff\\'] = _r[\\'my_active_hp\\'] - _r[\\'op_active_hp\\']\\n"
    "            _r[\\'energy_surplus\\'] = _r[\\'my_active_energy_count\\'] - _r[\\'remain_energy_cost\\']\\n"
    "            _r[\\'hand_advantage\\'] = _r[\\'my_hand_count\\'] - _r[\\'op_hand_count\\']\\n"
    "            _r[\\'bench_lead\\'] = _r[\\'my_bench_count\\'] - _r[\\'op_bench_count\\']\\n"
    "        prep = _U0_LGBM[\\'prep\\']"
)
NEW_INJ = (
    "        # v05d36+d39: add interaction and ratio features\\n"
    "        for _r in rows:\\n"
    "            _r[\\'can_attack\\'] = float(_r[\\'my_active_energy_count\\'] >= _r[\\'remain_energy_cost\\'])\\n"
    "            _r[\\'hp_diff\\'] = _r[\\'my_active_hp\\'] - _r[\\'op_active_hp\\']\\n"
    "            _r[\\'energy_surplus\\'] = _r[\\'my_active_energy_count\\'] - _r[\\'remain_energy_cost\\']\\n"
    "            _r[\\'hand_advantage\\'] = _r[\\'my_hand_count\\'] - _r[\\'op_hand_count\\']\\n"
    "            _r[\\'bench_lead\\'] = _r[\\'my_bench_count\\'] - _r[\\'op_bench_count\\']\\n"
    "            _r[\\'hp_ratio\\'] = _r[\\'my_active_hp\\'] / (_r[\\'my_active_hp\\'] + _r[\\'op_active_hp\\'] + 1.0)\\n"
    "            _r[\\'turn_normalized\\'] = min(_r[\\'turn\\'] / 30.0, 1.0)\\n"
    "            _r[\\'energy_ratio\\'] = _r[\\'my_active_energy_count\\'] / max(_r[\\'remain_energy_cost\\'], 1)\\n"
    "            _r[\\'deck_ratio\\'] = _r[\\'my_deck_count\\'] / (_r[\\'my_deck_count\\'] + _r[\\'op_deck_count\\'] + 1.0)\\n"
    "        prep = _U0_LGBM[\\'prep\\']"
)
assert OLD_INJ in src19, "OLD_INJ not found!"
src19 = src19.replace(OLD_INJ, NEW_INJ)

set_cell_src(cell19_idx, src19)
c19 = cell_src(cell19_idx)
assert 'hp_ratio' in c19 and 'turn_normalized' in c19 and 'feateng2' in c19
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
print(f"\nv05d39 notebook written: {DST_NB}")
