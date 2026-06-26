"""Create v05d26 from v05d25:
Expand training data to T200_OTHER episodes (T200 Alakazam vs non-T200 opponent).

Changes:
1. Cell 1: TOP200_EPISODE_FILTER = False (was True via env var default)
   -> DECISION_ROWS_DF gains T200_OTHER + OTHER_OTHER episodes
   -> prepare_alakazam_option_dataset's rank<=TOP_RANK_CUTOFF filter keeps
      only Top200 Alakazam players — so OTHER_OTHER and non-T200-Alakazam get excluded
   -> Net effect: +678 T200-Alakazam episodes from T200_OTHER (~+74% data)

2. Cell 19: Tier-aware asymmetric winner weighting
   T200_T200 winner=4x, T200_T200 loser=1x
   T200_OTHER winner=3x, T200_OTHER loser=0.5x
   Rationale: T200 losing to a weaker opponent is unusual (luck/blunder), trust less.

Data projection: ~442k → ~770k UNKNOWN_0 training rows
"""
import json, uuid
from pathlib import Path

WORK_DIR = Path('/kaggle/working')
SRC_NB = WORK_DIR / 'pokemon-20260625-v0-05d25.ipynb'
DST_NB = WORK_DIR / 'pokemon-20260625-v0-05d26.ipynb'

with open(SRC_NB) as f:
    nb = json.load(f)

def cell_src(i):
    return ''.join(nb['cells'][i]['source'])

def set_cell_src(i, s):
    nb['cells'][i]['source'] = s
    nb['cells'][i]['outputs'] = []
    if nb['cells'][i].get('cell_type') == 'code':
        nb['cells'][i]['execution_count'] = None

# ── Cell 1: version bump + disable TOP200_EPISODE_FILTER ─────────────────────
src1 = cell_src(1)
src1 = src1.replace("EXPERIMENT_NAME = 'v0_05d25_lgbm_winner_wt4'",
                    "EXPERIMENT_NAME = 'v0_05d26_lgbm_tier_aware'")
src1 = src1.replace("RUN_PREFIX = 'pokemon-20260625-v0-05d25'",
                    "RUN_PREFIX = 'pokemon-20260625-v0-05d26'")
# Override TOP200_EPISODE_FILTER: change env-var default from '1' to '0'
OLD_FILTER = "TOP200_EPISODE_FILTER = os.environ.get('V05_TOP200_FILTER', '1') != '0'"
NEW_FILTER = "TOP200_EPISODE_FILTER = os.environ.get('V05_TOP200_FILTER', '0') != '0'  # d26: default OFF to include T200_OTHER"
assert OLD_FILTER in src1, f"OLD_FILTER not found!\n{src1}"
src1 = src1.replace(OLD_FILTER, NEW_FILTER)
set_cell_src(1, src1)
assert 'v0_05d26_lgbm_tier_aware' in cell_src(1)
assert 'pokemon-20260625-v0-05d26' in cell_src(1)
assert "default OFF" in cell_src(1)
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

# ── Patch: replace flat winner_weight with tier-aware weighting ───────────────
OLD_WEIGHTING = (
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
NEW_WEIGHTING = (
    "            # Tier-aware winner weighting: T200_T200 winner=4x loser=1x, T200_OTHER winner=3x loser=0.5x\n"
    "            train_df = train_df.copy()\n"
    "            if 'won' in train_df.columns:\n"
    "                _is_winner = train_df['won'].fillna(0).astype(bool)\n"
    "                _is_t200t200 = (train_df.get('tier', pd.Series('T200_T200', index=train_df.index))\n"
    "                                .astype(str) == 'T200_T200')\n"
    "                train_df['_win_wt'] = (\n"
    "                    (_is_t200t200 &  _is_winner).astype(float) * 4.0 +\n"
    "                    (_is_t200t200 & ~_is_winner).astype(float) * 1.0 +\n"
    "                    (~_is_t200t200 &  _is_winner).astype(float) * 3.0 +\n"
    "                    (~_is_t200t200 & ~_is_winner).astype(float) * 0.5\n"
    "                )\n"
    "                _n_winner = int(_is_winner.sum())\n"
    "                _n_t200t200 = int(_is_t200t200.sum())\n"
    "                print(f'Tier-aware weighting: {_n_winner}/{len(train_df)} winner, {_n_t200t200} T200_T200 rows')\n"
    "            else:\n"
    "                train_df['_win_wt'] = 1.0\n"
    "                print('won column not found, using uniform weights')"
)
assert OLD_WEIGHTING in src19, "OLD_WEIGHTING not found in Cell 19!"
src19 = src19.replace(OLD_WEIGHTING, NEW_WEIGHTING)

# ── Update model name ─────────────────────────────────────────────────────────
OLD_MODEL_NAME = "                    'model': 'lightgbm_lambdarank_winner_wt4',"
NEW_MODEL_NAME = "                    'model': 'lightgbm_lambdarank_tier_aware',"
assert OLD_MODEL_NAME in src19, "OLD_MODEL_NAME not found!"
src19 = src19.replace(OLD_MODEL_NAME, NEW_MODEL_NAME)

set_cell_src(cell19_idx, src19)

c19 = cell_src(cell19_idx)
assert '_is_t200t200' in c19, "tier-aware weighting not in Cell 19!"
assert 'tier_aware' in c19, "model name not updated!"
print("Cell 19: OK")

# ── Add IDs / clear outputs ───────────────────────────────────────────────────
for cell in nb['cells']:
    if 'id' not in cell:
        cell['id'] = str(uuid.uuid4())[:8]
for cell in nb['cells']:
    if cell.get('cell_type') == 'code':
        cell['outputs'] = []
        cell['execution_count'] = None

with open(DST_NB, 'w') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print(f"\nv05d26 notebook written: {DST_NB}")
print(f"Total cells: {len(nb['cells'])}")
