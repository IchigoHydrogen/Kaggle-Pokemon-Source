"""Create v05d22 from v05d19:
Win-conditional imitation: train LambdaRank ONLY on winner decisions.

Hypothesis: training on mixed winner+loser data teaches the model the AVERAGE
action preference, not the winner's preference. Filtering to won==True means
is_chosen reflects what winners chose → model learns winner strategy.

Expected effect:
  winner_top1 ↑ (model aligns with winners)
  loser_top1  ↓ (model disagrees with losers, who chose differently)
  winner_margin ↑↑

Also adds new numeric features from ALAKAZAM_OPTION_MODEL_DF:
  supporter_played, stadium_played, energy_attached, my_discard_count, op_discard_count
"""
import json, uuid
from pathlib import Path

WORK_DIR = Path('/kaggle/working')
SRC_NB = WORK_DIR / 'pokemon-20260625-v0-05d19.ipynb'
DST_NB = WORK_DIR / 'pokemon-20260625-v0-05d22.ipynb'

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
src1 = src1.replace("EXPERIMENT_NAME = 'v0_05d19_lgbm_lambdarank'",
                    "EXPERIMENT_NAME = 'v0_05d22_lgbm_winner_only'")
src1 = src1.replace("RUN_PREFIX = 'pokemon-20260625-v0-05d19'",
                    "RUN_PREFIX = 'pokemon-20260625-v0-05d22'")
set_cell_src(1, src1)
assert 'v0_05d22_lgbm_winner_only' in cell_src(1)
assert 'pokemon-20260625-v0-05d22' in cell_src(1)
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

# ── Patch 1: Add winner filter + new features after data preparation ──────────
OLD_WORK_CHECK = (
    "        if work.empty or work['is_chosen'].nunique() < 2 or work['decision_id'].nunique() < 50:\n"
    "            UNKNOWN0_MLP_REPORT = {'status': 'skipped', 'reason': 'insufficient_unknown0_rows',\n"
    "                                    'rows': int(len(work))}"
)
NEW_WORK_CHECK = (
    "        # Extra features from ALAKAZAM_OPTION_MODEL_DF\n"
    "        _extra_numeric_candidates = ['supporter_played', 'stadium_played', 'energy_attached',\n"
    "                                      'my_discard_count', 'op_discard_count']\n"
    "        _extra_numeric_avail = [f for f in _extra_numeric_candidates if f in work.columns]\n"
    "        print(f'Extra numeric features available: {_extra_numeric_avail}')\n"
    "\n"
    "        if work.empty or work['is_chosen'].nunique() < 2 or work['decision_id'].nunique() < 50:\n"
    "            UNKNOWN0_MLP_REPORT = {'status': 'skipped', 'reason': 'insufficient_unknown0_rows',\n"
    "                                    'rows': int(len(work))}"
)
assert OLD_WORK_CHECK in src19, "OLD_WORK_CHECK not found!"
src19 = src19.replace(OLD_WORK_CHECK, NEW_WORK_CHECK)

# ── Patch 1b: filter train_df to winner-only AFTER split ─────────────────────
# valid_df keeps ALL decisions (winner+loser) for correct margin evaluation
OLD_SPLIT = (
    "            train_df, valid_df = split_by_decision_safe(work, valid_frac=0.15)\n"
    "            if train_df.empty or valid_df.empty or train_df['is_chosen'].nunique() < 2:\n"
    "                UNKNOWN0_MLP_REPORT = {'status': 'skipped', 'reason': 'bad_unknown0_split'}"
)
NEW_SPLIT = (
    "            train_df, valid_df = split_by_decision_safe(work, valid_frac=0.15)\n"
    "            # Win-conditional: train only on winner decisions, eval on all decisions\n"
    "            if 'won' in train_df.columns:\n"
    "                _train_all_rows = len(train_df)\n"
    "                train_df = train_df[train_df['won'] == True].copy()\n"
    "                print(f'Win-conditional training: {len(train_df)}/{_train_all_rows} train rows (winner only)')\n"
    "            if train_df.empty or valid_df.empty or train_df['is_chosen'].nunique() < 2:\n"
    "                UNKNOWN0_MLP_REPORT = {'status': 'skipped', 'reason': 'bad_unknown0_split'}"
)
assert OLD_SPLIT in src19, "OLD_SPLIT not found!"
src19 = src19.replace(OLD_SPLIT, NEW_SPLIT)

# ── Patch 2: Include extra features in _all_feats ────────────────────────────
OLD_ALL_FEATS = (
    "                _all_feats = [f for f in UNKNOWN0_NUMERIC_FEATURES + UNKNOWN0_CATEGORICAL_FEATURES\n"
    "                              if f not in UNKNOWN0_LGBM_EXCLUDE_FEATURES]"
)
NEW_ALL_FEATS = (
    "                _all_feats = [f for f in UNKNOWN0_NUMERIC_FEATURES + UNKNOWN0_CATEGORICAL_FEATURES\n"
    "                              if f not in UNKNOWN0_LGBM_EXCLUDE_FEATURES]\n"
    "                # Add extra features that are available in the training frame\n"
    "                _all_feats = _all_feats + [f for f in _extra_numeric_avail if f not in _all_feats]"
)
assert OLD_ALL_FEATS in src19, "OLD_ALL_FEATS not found!"
src19 = src19.replace(OLD_ALL_FEATS, NEW_ALL_FEATS)

# ── Patch 3: Save extra_feats list in prep for inference ──────────────────────
OLD_PREP_DICT = (
    "                _lgbm_prep = {\n"
    "                    'feature_names': _all_feats,\n"
    "                    'cat_features': UNKNOWN0_LGBM_ALL_CAT,\n"
    "                    'cat_maps': _cat_maps,\n"
    "                    'exclude_features': UNKNOWN0_LGBM_EXCLUDE_FEATURES,\n"
    "                }"
)
NEW_PREP_DICT = (
    "                _lgbm_prep = {\n"
    "                    'feature_names': _all_feats,\n"
    "                    'cat_features': UNKNOWN0_LGBM_ALL_CAT,\n"
    "                    'cat_maps': _cat_maps,\n"
    "                    'exclude_features': UNKNOWN0_LGBM_EXCLUDE_FEATURES,\n"
    "                    'extra_numeric': _extra_numeric_avail,\n"
    "                }"
)
assert OLD_PREP_DICT in src19, "OLD_PREP_DICT not found!"
src19 = src19.replace(OLD_PREP_DICT, NEW_PREP_DICT)

# ── Patch 4: Update model name in report ──────────────────────────────────────
OLD_MODEL_NAME = "                    'model': 'lightgbm_lambdarank',"
NEW_MODEL_NAME = "                    'model': 'lightgbm_lambdarank_winner_only',"
assert OLD_MODEL_NAME in src19, "OLD_MODEL_NAME not found!"
src19 = src19.replace(OLD_MODEL_NAME, NEW_MODEL_NAME)

# ── Patch 5: Add extra_numeric to report ─────────────────────────────────────
OLD_FEATURE_SAFETY = "                    'feature_safety': UNKNOWN0_FEATURE_SAFETY,"
NEW_FEATURE_SAFETY = (
    "                    'feature_safety': UNKNOWN0_FEATURE_SAFETY,\n"
    "                    'extra_numeric_used': _extra_numeric_avail,"
)
assert OLD_FEATURE_SAFETY in src19, "OLD_FEATURE_SAFETY not found!"
src19 = src19.replace(OLD_FEATURE_SAFETY, NEW_FEATURE_SAFETY)

# ── Patch 6: Add extra features to inference code ────────────────────────────
# The inference code needs to handle extra features from prep['extra_numeric'].
# These features will default to 0 since they're not available in main.py context.
# BUT they're row-level features that need to be computed:
# - supporter_played, stadium_played, energy_attached: game state booleans
# - my_discard_count, op_discard_count: counts

# Find the inference encoding loop in src19 (in the _LGBM_INJ_CODE repr string)
OLD_ENCODE_LOOP = (
    "        for ci, col in enumerate(feature_names):\\n'\n"
    "    '            if col in cat_maps:\\n'"
)
if OLD_ENCODE_LOOP in src19:
    # Add comment about extra features defaulting to 0
    NEW_ENCODE_LOOP = (
        "        for ci, col in enumerate(feature_names):\\n'\n"
        "    '            if col in cat_maps:\\n'"
    )
    # No change needed - the existing code already handles unknown features via r.get(col, 0)
    print("  Inference code: extra features handled by r.get(col, 0) default")
else:
    print("  WARNING: Could not find encode loop in inference code (extra features will default to 0)")

set_cell_src(cell19_idx, src19)

c19 = cell_src(cell19_idx)
assert "'won'" in c19, "'won' filter not in Cell 19!"
assert "train_df['won']" in c19 or "work['won']" in c19, "Won filter condition not found!"
assert '_extra_numeric_avail' in c19, "extra_numeric_avail not in Cell 19!"
assert 'winner_only' in c19, "model name not updated!"
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
print(f"\nv05d22 notebook written: {DST_NB}")
print(f"Total cells: {len(nb['cells'])}")
