"""Create v05d24 from v05d22:
Winner-weighted training instead of winner-ONLY filter.

Hypothesis: hard winner-only filter loses 44% of training data (loser decisions).
Soft weighting (winner=2x, loser=1x) preserves all data while still prioritizing
winner patterns. This might capture the best of both worlds.

Also try a smaller model (num_leaves=63) since d22's 127-leaf model with only 47
best_iter might be overfitting with winner-only filtered data.
"""
import json, uuid
from pathlib import Path

WORK_DIR = Path('/kaggle/working')
SRC_NB = WORK_DIR / 'pokemon-20260625-v0-05d22.ipynb'
DST_NB = WORK_DIR / 'pokemon-20260625-v0-05d24.ipynb'

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
src1 = src1.replace("EXPERIMENT_NAME = 'v0_05d22_lgbm_winner_only'",
                    "EXPERIMENT_NAME = 'v0_05d24_lgbm_winner_wt2'")
src1 = src1.replace("RUN_PREFIX = 'pokemon-20260625-v0-05d22'",
                    "RUN_PREFIX = 'pokemon-20260625-v0-05d24'")
set_cell_src(1, src1)
assert 'v0_05d24_lgbm_winner_wt2' in cell_src(1)
assert 'pokemon-20260625-v0-05d24' in cell_src(1)
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

# ── Patch 1: replace hard winner filter with soft weighting ──────────────────
OLD_WINNER_FILTER = (
    "            # Win-conditional: train only on winner decisions, eval on all decisions\n"
    "            if 'won' in train_df.columns:\n"
    "                _train_all_rows = len(train_df)\n"
    "                train_df = train_df[train_df['won'] == True].copy()\n"
    "                print(f'Win-conditional training: {len(train_df)}/{_train_all_rows} train rows (winner only)')"
)
NEW_WINNER_FILTER = (
    "            # Winner-weighted training: winner decisions get 2x weight\n"
    "            if 'won' in train_df.columns:\n"
    "                _winner_weight = 2.0\n"
    "                train_df = train_df.copy()\n"
    "                train_df['_win_wt'] = train_df['won'].fillna(0).astype(float) * (_winner_weight - 1.0) + 1.0\n"
    "                _n_winner = int((train_df['won'] == True).sum())\n"
    "                print(f'Winner-weighted training: {_n_winner}/{len(train_df)} winner rows (weight={_winner_weight}x)')\n"
    "            else:\n"
    "                train_df = train_df.copy()\n"
    "                train_df['_win_wt'] = 1.0\n"
    "                print('won column not found, using uniform weights')"
)
assert OLD_WINNER_FILTER in src19, "OLD_WINNER_FILTER not found!"
src19 = src19.replace(OLD_WINNER_FILTER, NEW_WINNER_FILTER)

# ── Patch 2: use _win_wt as sample weight in LambdaRank ──────────────────────
OLD_DATASET = (
    "                _cat_cols = [c for c in UNKNOWN0_LGBM_ALL_CAT if c in X_tr.columns]\n"
    "                _dtr = _lgb_mod.Dataset(\n"
    "                    X_tr, label=y_tr, group=_g_tr,\n"
    "                    categorical_feature=_cat_cols, free_raw_data=False)\n"
    "                _dva = _lgb_mod.Dataset(\n"
    "                    X_va, label=y_va, group=_g_va, reference=_dtr,\n"
    "                    categorical_feature=_cat_cols)"
)
NEW_DATASET = (
    "                # Aggregate win weights to group level (mean per decision)\n"
    "                _w_wt_tr = (train_df.groupby('decision_id', sort=False)['_win_wt']\n"
    "                            .transform('mean').values.astype(float)\n"
    "                            if '_win_wt' in train_df.columns else None)\n"
    "                _cat_cols = [c for c in UNKNOWN0_LGBM_ALL_CAT if c in X_tr.columns]\n"
    "                _dtr = _lgb_mod.Dataset(\n"
    "                    X_tr, label=y_tr, group=_g_tr, weight=_w_wt_tr,\n"
    "                    categorical_feature=_cat_cols, free_raw_data=False)\n"
    "                _dva = _lgb_mod.Dataset(\n"
    "                    X_va, label=y_va, group=_g_va, reference=_dtr,\n"
    "                    categorical_feature=_cat_cols)"
)
assert OLD_DATASET in src19, "OLD_DATASET not found!"
src19 = src19.replace(OLD_DATASET, NEW_DATASET)

# ── Patch 3: update model name and add winner_weight to report ────────────────
OLD_MODEL_NAME = "                    'model': 'lightgbm_lambdarank_winner_only',"
NEW_MODEL_NAME = "                    'model': 'lightgbm_lambdarank_winner_wt2',"
assert OLD_MODEL_NAME in src19, "OLD_MODEL_NAME not found!"
src19 = src19.replace(OLD_MODEL_NAME, NEW_MODEL_NAME)

# ── Patch 4: num_leaves 127 → 63 (try smaller model for better generalization) ──
OLD_LEAVES = "                    'num_leaves': 127,"
NEW_LEAVES = "                    'num_leaves': 63,"
assert OLD_LEAVES in src19, "OLD_LEAVES not found!"
src19 = src19.replace(OLD_LEAVES, NEW_LEAVES)

set_cell_src(cell19_idx, src19)

c19 = cell_src(cell19_idx)
assert '_win_wt' in c19, "_win_wt not in Cell 19!"
assert 'winner_wt' in c19 or 'winner_weight' in c19, "winner weight not in Cell 19!"
assert "'num_leaves': 63" in c19, "num_leaves not updated!"
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
print(f"\nv05d24 notebook written: {DST_NB}")
print(f"Total cells: {len(nb['cells'])}")
