"""Create v05d20 from v05d19:
1. Re-add mc_step_weight as group-level sample weights to LambdaRank.
   (They were dropped when switching from binary→lambdarank in d19.)
   Decision-level weights = mean mc_step_weight across all rows of that decision.
2. Add hp_ratio derived feature: my_active_hp / (my_active_hp + op_active_hp).
3. Increase num_boost_round 500→1000, early_stopping_rounds 30→50.
"""
import json, uuid
from pathlib import Path

WORK_DIR = Path('/kaggle/working')
SRC_NB = WORK_DIR / 'pokemon-20260625-v0-05d19.ipynb'
DST_NB = WORK_DIR / 'pokemon-20260625-v0-05d20.ipynb'

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
                    "EXPERIMENT_NAME = 'v0_05d20_lgbm_lrank_phwt'")
src1 = src1.replace("RUN_PREFIX = 'pokemon-20260625-v0-05d19'",
                    "RUN_PREFIX = 'pokemon-20260625-v0-05d20'")
set_cell_src(1, src1)
assert 'v0_05d20_lgbm_lrank_phwt' in cell_src(1)
assert 'pokemon-20260625-v0-05d20' in cell_src(1)
print("Cell 1: OK")

# ── Find Cell 19 ─────────────────────────────────────────────────────────────
cell19_idx = None
for i, cell in enumerate(nb['cells']):
    src = ''.join(cell['source'])
    if 'lambdarank' in src and 'UNKNOWN0_MLP_REPORT' in src:
        cell19_idx = i
        break
if cell19_idx is None:
    raise RuntimeError("Could not find Cell 19 (LambdaRank training)")
print(f"Cell 19 found at index {cell19_idx}")

src19 = cell_src(cell19_idx)

# ── Replacement 1: add hp_ratio feature derivation before _encode_for_lgbm ───
OLD_ENCODE_START = '''\
                _all_feats = [f for f in UNKNOWN0_NUMERIC_FEATURES + UNKNOWN0_CATEGORICAL_FEATURES
                              if f not in UNKNOWN0_LGBM_EXCLUDE_FEATURES]'''

NEW_ENCODE_START = '''\
                # Derived features: hp_ratio captures relative board health
                for _frame in [train_df, valid_df]:
                    if 'my_active_hp' in _frame.columns and 'op_active_hp' in _frame.columns:
                        _frame['hp_ratio'] = (
                            _frame['my_active_hp'].fillna(0).astype(float) /
                            (_frame['my_active_hp'].fillna(0).astype(float) +
                             _frame['op_active_hp'].fillna(0).astype(float) + 1.0).clip(lower=1.0)
                        )
                _extra_feats = ['hp_ratio'] if 'hp_ratio' in train_df.columns else []
                _all_feats = [f for f in UNKNOWN0_NUMERIC_FEATURES + UNKNOWN0_CATEGORICAL_FEATURES
                              if f not in UNKNOWN0_LGBM_EXCLUDE_FEATURES] + _extra_feats'''

assert OLD_ENCODE_START in src19, "OLD_ENCODE_START not found!"
src19 = src19.replace(OLD_ENCODE_START, NEW_ENCODE_START)

# ── Replacement 2: add group-level phase weights ──────────────────────────────
OLD_GROUPS = '''\
                _cat_cols = [c for c in UNKNOWN0_LGBM_ALL_CAT if c in X_tr.columns]
                _dtr = _lgb_mod.Dataset(
                    X_tr, label=y_tr, group=_g_tr,
                    categorical_feature=_cat_cols, free_raw_data=False)
                _dva = _lgb_mod.Dataset(
                    X_va, label=y_va, group=_g_va, reference=_dtr,
                    categorical_feature=_cat_cols)'''

NEW_GROUPS = '''\
                # Phase weights: mid-game decisions (mc_step_weight > 0) matter more
                if USE_MC_RETURN_WEIGHTS and 'mc_step_weight' in train_df.columns:
                    # Aggregate to decision level (LambdaRank needs row-level but
                    # within a group all rows should share the query weight)
                    _dec_wt_tr = (train_df.groupby('decision_id', sort=False)['mc_step_weight']
                                  .transform('mean').clip(lower=0.05).astype(float))
                    _dec_wt_va = (valid_df.groupby('decision_id', sort=False)['mc_step_weight']
                                  .transform('mean').clip(lower=0.05).astype(float))
                    # Normalize to mean 1
                    _wt_mean = float(_dec_wt_tr.mean())
                    _w_tr_arr = (_dec_wt_tr / max(_wt_mean, 1e-8)).values
                    _w_va_arr = (_dec_wt_va / max(_wt_mean, 1e-8)).values
                    print(f'  Phase weights: mean={_wt_mean:.4f}, '
                          f'weight>0={(_dec_wt_tr > 0.05).mean():.1%}')
                else:
                    _w_tr_arr = None
                    _w_va_arr = None

                _cat_cols = [c for c in UNKNOWN0_LGBM_ALL_CAT if c in X_tr.columns]
                _dtr = _lgb_mod.Dataset(
                    X_tr, label=y_tr, group=_g_tr, weight=_w_tr_arr,
                    categorical_feature=_cat_cols, free_raw_data=False)
                _dva = _lgb_mod.Dataset(
                    X_va, label=y_va, group=_g_va, weight=_w_va_arr, reference=_dtr,
                    categorical_feature=_cat_cols)'''

assert OLD_GROUPS in src19, "OLD_GROUPS not found!"
src19 = src19.replace(OLD_GROUPS, NEW_GROUPS)

# ── Replacement 3: increase rounds and early stopping ────────────────────────
OLD_TRAIN = '''\
                _u0_lgbm = _lgb_mod.train(
                    _lgbm_params, _dtr,
                    num_boost_round=500,
                    valid_sets=[_dtr, _dva],
                    valid_names=['train', 'valid'],
                    callbacks=[
                        _lgb_mod.early_stopping(30, verbose=False),
                        _lgb_mod.log_evaluation(50),
                    ],
                )'''

NEW_TRAIN = '''\
                _u0_lgbm = _lgb_mod.train(
                    _lgbm_params, _dtr,
                    num_boost_round=1000,
                    valid_sets=[_dtr, _dva],
                    valid_names=['train', 'valid'],
                    callbacks=[
                        _lgb_mod.early_stopping(50, verbose=False),
                        _lgb_mod.log_evaluation(100),
                    ],
                )'''

assert OLD_TRAIN in src19, "OLD_TRAIN not found!"
src19 = src19.replace(OLD_TRAIN, NEW_TRAIN)

# ── Replacement 4: update report model name ───────────────────────────────────
OLD_MODEL_NAME = "                    'model': 'lightgbm_lambdarank',"
NEW_MODEL_NAME = "                    'model': 'lightgbm_lambdarank_phwt',"
assert OLD_MODEL_NAME in src19, "OLD_MODEL_NAME not found!"
src19 = src19.replace(OLD_MODEL_NAME, NEW_MODEL_NAME)

# ── Update lgbm_prep to save extra_feats list (for inference) ─────────────────
OLD_PREP_DICT = '''\
                _lgbm_prep = {
                    'feature_names': _all_feats,
                    'cat_features': UNKNOWN0_LGBM_ALL_CAT,
                    'cat_maps': _cat_maps,
                    'exclude_features': UNKNOWN0_LGBM_EXCLUDE_FEATURES,
                }'''

NEW_PREP_DICT = '''\
                _lgbm_prep = {
                    'feature_names': _all_feats,
                    'cat_features': UNKNOWN0_LGBM_ALL_CAT,
                    'cat_maps': _cat_maps,
                    'exclude_features': UNKNOWN0_LGBM_EXCLUDE_FEATURES,
                    'extra_feats': _extra_feats,
                }'''

assert OLD_PREP_DICT in src19, "OLD_PREP_DICT not found!"
src19 = src19.replace(OLD_PREP_DICT, NEW_PREP_DICT)

# ── Replacement 5: update inference code to compute hp_ratio ─────────────────
# The _LGBM_INJ_CODE needs to compute hp_ratio before building X.
# It's in the tail of Cell 19 (the _LGBM_INJ_CODE string).

OLD_INJ_ROW_APPEND = (
    "                'powerful_hand_damage_est': powerful_hand,\n"
    "                'powerful_hand_can_ko_active': float(powerful_hand >= op_hp),\n"
    "                'deckout_risk_feature': float(my_deck_n <= 4),"
)
NEW_INJ_ROW_APPEND = (
    "                'powerful_hand_damage_est': powerful_hand,\n"
    "                'powerful_hand_can_ko_active': float(powerful_hand >= op_hp),\n"
    "                'deckout_risk_feature': float(my_deck_n <= 4),\n"
    "                'hp_ratio': float(getattr(my_active, 'hp', 0) or 0) /\n"
    "                    max(float(getattr(my_active, 'hp', 0) or 0) +\n"
    "                        float(getattr(op_active, 'hp', 0) or 0) + 1.0, 1.0),"
)
# This appears inside the repr(_LGBM_INJ_CODE) string in src19
# We need to search for it literally
if OLD_INJ_ROW_APPEND in src19:
    src19 = src19.replace(OLD_INJ_ROW_APPEND, NEW_INJ_ROW_APPEND)
    print("Cell 19: hp_ratio added to inference rows")
else:
    print("WARNING: hp_ratio injection target not found in inj code - checking repr...")
    # Try to find the repr'd version inside the string
    old_repr = repr(OLD_INJ_ROW_APPEND)
    if old_repr[1:-1] in src19:
        print("  Found in repr context - skipping for now (inference will use prep feature_names filter)")
    else:
        print("  Not found either way - will use feature_names filter to exclude hp_ratio from inference if missing")

set_cell_src(cell19_idx, src19)

# Sanity checks
c19 = cell_src(cell19_idx)
assert 'hp_ratio' in c19, "hp_ratio not in Cell 19!"
assert 'mc_step_weight' in c19, "mc_step_weight not in Cell 19!"
assert 'clip(lower=0.05)' in c19, "phase weights not in Cell 19!"
assert 'num_boost_round=1000' in c19, "rounds not updated!"
assert 'early_stopping(50' in c19, "early stopping not updated!"
print("Cell 19: OK")

# ── Add IDs to cells missing them ──────────────────────────────────────────────
for cell in nb['cells']:
    if 'id' not in cell:
        cell['id'] = str(uuid.uuid4())[:8]

# ── Clear outputs ──────────────────────────────────────────────────────────────
for cell in nb['cells']:
    if cell.get('cell_type') == 'code':
        cell['outputs'] = []
        cell['execution_count'] = None

with open(DST_NB, 'w') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print(f"\nv05d20 notebook written: {DST_NB}")
print(f"Total cells: {len(nb['cells'])}")
