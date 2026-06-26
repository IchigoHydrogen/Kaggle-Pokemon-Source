"""Create v05d29 from v05d25:
Replace flat winner_weight with graded LambdaRank labels.

Instead of: binary {0,1} labels + 4x winner query weight
Use:        graded {0,1,2} labels, no winner_weight
  - winner's chosen option → label=2
  - loser's chosen option  → label=1
  - not chosen             → label=0

Rationale: graded labels encode winner/loser priority directly in NDCG
objective rather than through query-level scaling. NDCG with labels [0,1,2]
naturally penalizes ranking a loser's choice above a winner's choice more
than ranking unchosen above chosen.

Keep max_rows=350k, num_leaves=63, early_stopping=50 (d25 baseline config).
"""
import json, uuid
from pathlib import Path

WORK_DIR = Path('/kaggle/working')
SRC_NB = WORK_DIR / 'pokemon-20260625-v0-05d25.ipynb'
DST_NB = WORK_DIR / 'pokemon-20260625-v0-05d29.ipynb'

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
                    "EXPERIMENT_NAME = 'v0_05d29_lgbm_graded_label'")
src1 = src1.replace("RUN_PREFIX = 'pokemon-20260625-v0-05d25'",
                    "RUN_PREFIX = 'pokemon-20260625-v0-05d29'")
set_cell_src(1, src1)
assert 'v0_05d29_lgbm_graded_label' in cell_src(1)
assert 'pokemon-20260625-v0-05d29' in cell_src(1)
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

# ── Patch 1: replace winner_weight block with graded-label announcement ───────
OLD_WINNER_BLOCK = (
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
NEW_WINNER_BLOCK = (
    "            # Graded labels: winner chosen=2, loser chosen=1, unchosen=0 (no winner_weight)\n"
    "            train_df = train_df.copy()\n"
    "            if 'won' in train_df.columns:\n"
    "                _n_winner = int((train_df['won'] == True).sum())\n"
    "                print(f'Graded labels: {_n_winner}/{len(train_df)} winner rows → label=2; loser → label=1')\n"
    "            else:\n"
    "                print('won column not found, using binary labels')"
)
assert OLD_WINNER_BLOCK in src19, "OLD_WINNER_BLOCK not found!"
src19 = src19.replace(OLD_WINNER_BLOCK, NEW_WINNER_BLOCK)

# ── Patch 2: graded y_tr/y_va (winner_chosen=2, loser_chosen=1, unchosen=0) ──
OLD_LABELS = (
    "                y_tr = train_df['is_chosen'].astype(int)\n"
    "                y_va = valid_df['is_chosen'].astype(int)"
)
NEW_LABELS = (
    "                # Graded relevance: winner chosen=2, loser chosen=1, unchosen=0\n"
    "                if 'won' in train_df.columns:\n"
    "                    y_tr = (train_df['is_chosen'].astype(int) *\n"
    "                            (train_df['won'].fillna(0).astype(int) + 1))\n"
    "                else:\n"
    "                    y_tr = train_df['is_chosen'].astype(int)\n"
    "                if 'won' in valid_df.columns:\n"
    "                    y_va = (valid_df['is_chosen'].astype(int) *\n"
    "                            (valid_df['won'].fillna(0).astype(int) + 1))\n"
    "                else:\n"
    "                    y_va = valid_df['is_chosen'].astype(int)"
)
assert OLD_LABELS in src19, "OLD_LABELS not found!"
src19 = src19.replace(OLD_LABELS, NEW_LABELS)

# ── Patch 3: remove _w_wt_tr and weight= param from Dataset ──────────────────
OLD_DATASET = (
    "                # Aggregate win weights to group level (mean per decision)\n"
    "                _w_wt_tr = (train_df.groupby('decision_id', sort=False)['_win_wt']\n"
    "                            .transform('mean').values.astype(float)\n"
    "                            if '_win_wt' in train_df.columns else None)\n"
    "                _cat_cols = [c for c in UNKNOWN0_LGBM_ALL_CAT if c in X_tr.columns]\n"
    "                _dtr = _lgb_mod.Dataset(\n"
    "                    X_tr, label=y_tr, group=_g_tr, weight=_w_wt_tr,\n"
    "                    categorical_feature=_cat_cols, free_raw_data=False)"
)
NEW_DATASET = (
    "                _cat_cols = [c for c in UNKNOWN0_LGBM_ALL_CAT if c in X_tr.columns]\n"
    "                _dtr = _lgb_mod.Dataset(\n"
    "                    X_tr, label=y_tr, group=_g_tr,\n"
    "                    categorical_feature=_cat_cols, free_raw_data=False)"
)
assert OLD_DATASET in src19, "OLD_DATASET not found!"
src19 = src19.replace(OLD_DATASET, NEW_DATASET)

# ── Patch 4: model name ───────────────────────────────────────────────────────
OLD_MODEL_NAME = "                    'model': 'lightgbm_lambdarank_winner_wt4',"
NEW_MODEL_NAME = "                    'model': 'lightgbm_lambdarank_graded_label',"
assert OLD_MODEL_NAME in src19, "OLD_MODEL_NAME not found!"
src19 = src19.replace(OLD_MODEL_NAME, NEW_MODEL_NAME)

set_cell_src(cell19_idx, src19)

c19 = cell_src(cell19_idx)
assert 'graded_label' in c19, "model name not updated!"
assert "fillna(0).astype(int) + 1)" in c19, "graded labels not in Cell 19!"
assert '_w_wt_tr' not in c19, "_w_wt_tr should be removed!"
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
print(f"\nv05d29 notebook written: {DST_NB}")
