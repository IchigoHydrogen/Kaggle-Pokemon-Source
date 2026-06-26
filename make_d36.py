"""Create v05d36 from v05d25:
Feature engineering — add interaction features:

  can_attack          = (my_active_energy_count >= remain_energy_cost)
  hp_diff             = my_active_hp - op_active_hp
  energy_surplus      = my_active_energy_count - remain_energy_cost
  hand_advantage      = my_hand_count - op_hand_count
  bench_lead          = my_bench_count - op_bench_count

These capture key "who's winning the board" signals that trees don't
naturally find via feature pairs. All computable at inference time.

Winner_weight=4x unchanged (d25 base).
"""
import json, uuid
from pathlib import Path

WORK_DIR = Path('/kaggle/working')
SRC_NB = WORK_DIR / 'pokemon-20260625-v0-05d25.ipynb'
DST_NB = WORK_DIR / 'pokemon-20260625-v0-05d36.ipynb'

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
                    "EXPERIMENT_NAME = 'v0_05d36_lgbm_feateng'")
src1 = src1.replace("RUN_PREFIX = 'pokemon-20260625-v0-05d25'",
                    "RUN_PREFIX = 'pokemon-20260625-v0-05d36'")
set_cell_src(1, src1)
assert 'v0_05d36_lgbm_feateng' in cell_src(1)
print("Cell 1: OK")

cell19_idx = None
for i, cell in enumerate(nb['cells']):
    src = ''.join(cell['source'])
    if 'lambdarank' in src and 'UNKNOWN0_MLP_REPORT' in src:
        cell19_idx = i
        break
assert cell19_idx is not None
src19 = cell_src(cell19_idx)

# ── Patch 1: add new features to UNKNOWN0_NUMERIC_FEATURES ───────────────────
OLD_FEAT = (
    "    'my_prizes_left', 'op_prizes_left', 'stadium_id', 'powerful_hand_damage_est',\n"
    "    'powerful_hand_can_ko_active', 'deckout_risk_feature',\n"
    "]"
)
NEW_FEAT = (
    "    'my_prizes_left', 'op_prizes_left', 'stadium_id', 'powerful_hand_damage_est',\n"
    "    'powerful_hand_can_ko_active', 'deckout_risk_feature',\n"
    "    # v05d36: interaction features\n"
    "    'can_attack', 'hp_diff', 'energy_surplus', 'hand_advantage', 'bench_lead',\n"
    "]"
)
assert OLD_FEAT in src19, f"OLD_FEAT not found!"
src19 = src19.replace(OLD_FEAT, NEW_FEAT)

# ── Patch 2: compute new features in train_df / valid_df before training ─────
OLD_PREP = (
    "            # Winner-weighted training: winner decisions get 2x weight\n"
)
NEW_PREP = (
    "            # v05d36: compute interaction features\n"
    "            for _fdf in [train_df, valid_df]:\n"
    "                _fdf['can_attack'] = (_fdf['my_active_energy_count'] >= _fdf['remain_energy_cost']).astype(float)\n"
    "                _fdf['hp_diff'] = _fdf['my_active_hp'] - _fdf['op_active_hp']\n"
    "                _fdf['energy_surplus'] = _fdf['my_active_energy_count'] - _fdf['remain_energy_cost']\n"
    "                _fdf['hand_advantage'] = _fdf['my_hand_count'] - _fdf['op_hand_count']\n"
    "                _fdf['bench_lead'] = _fdf['my_bench_count'] - _fdf['op_bench_count']\n"
    "            print('Feature engineering: can_attack, hp_diff, energy_surplus, hand_advantage, bench_lead')\n"
    "            # Winner-weighted training: winner decisions get 2x weight\n"
)
assert OLD_PREP in src19, "OLD_PREP not found!"
src19 = src19.replace(OLD_PREP, NEW_PREP)

# ── Patch 3: model name ───────────────────────────────────────────────────────
OLD_MODEL_NAME = "                    'model': 'lightgbm_lambdarank_winner_wt4',"
NEW_MODEL_NAME = "                    'model': 'lightgbm_lambdarank_feateng',"
assert OLD_MODEL_NAME in src19
src19 = src19.replace(OLD_MODEL_NAME, NEW_MODEL_NAME)

# ── Patch 4: update _LGBM_INJ_CODE to compute new features ───────────────────
# The INJ_CODE is a repr'd string — search for escaped version
OLD_INJ = (
    "\\n                \\'option_signature\\': sig,\\n"
    "            })\\n"
    "        prep = _U0_LGBM[\\'prep\\']"
)
NEW_INJ = (
    "\\n                \\'option_signature\\': sig,\\n"
    "            })\\n"
    "        # v05d36: add interaction features\\n"
    "        for _r in rows:\\n"
    "            _r[\\'can_attack\\'] = float(_r[\\'my_active_energy_count\\'] >= _r[\\'remain_energy_cost\\'])\\n"
    "            _r[\\'hp_diff\\'] = _r[\\'my_active_hp\\'] - _r[\\'op_active_hp\\']\\n"
    "            _r[\\'energy_surplus\\'] = _r[\\'my_active_energy_count\\'] - _r[\\'remain_energy_cost\\']\\n"
    "            _r[\\'hand_advantage\\'] = _r[\\'my_hand_count\\'] - _r[\\'op_hand_count\\']\\n"
    "            _r[\\'bench_lead\\'] = _r[\\'my_bench_count\\'] - _r[\\'op_bench_count\\']\\n"
    "        prep = _U0_LGBM[\\'prep\\']"
)
assert OLD_INJ in src19, f"OLD_INJ not found! Searching: {repr(OLD_INJ[:50])}"
src19 = src19.replace(OLD_INJ, NEW_INJ)

set_cell_src(cell19_idx, src19)
c19 = cell_src(cell19_idx)
assert 'can_attack' in c19 and 'hp_diff' in c19 and 'feateng' in c19
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
print(f"\nv05d36 notebook written: {DST_NB}")
