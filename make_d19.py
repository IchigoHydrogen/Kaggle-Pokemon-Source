"""Create v05d19 from v05d18:
1. Switch LightGBM objective from binary → lambdarank (group by decision_id).
   Ranking directly optimizes the "pick the best option" task, avoiding
   class-imbalance hacks like scale_pos_weight.
2. Exclude 'turn_action_count' from LGBM features — it's always 0.0 in
   inference but varies in training (train/test mismatch).
3. Increase num_leaves 63 → 127 and set more aggressive feature/bagging fractions.
"""
import json, uuid
from pathlib import Path

WORK_DIR = Path('/kaggle/working')
SRC_NB = WORK_DIR / 'pokemon-20260625-v0-05d18.ipynb'
DST_NB = WORK_DIR / 'pokemon-20260625-v0-05d19.ipynb'

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
src1 = src1.replace("EXPERIMENT_NAME = 'v0_05d18_lgbm_cat_id'",
                    "EXPERIMENT_NAME = 'v0_05d19_lgbm_lambdarank'")
src1 = src1.replace("RUN_PREFIX = 'pokemon-20260625-v0-05d18'",
                    "RUN_PREFIX = 'pokemon-20260625-v0-05d19'")
set_cell_src(1, src1)
assert 'v0_05d19_lgbm_lambdarank' in cell_src(1)
assert 'pokemon-20260625-v0-05d19' in cell_src(1)
print("Cell 1: OK")

# ── Find Cell 19 ─────────────────────────────────────────────────────────────
cell19_idx = None
for i, cell in enumerate(nb['cells']):
    src = ''.join(cell['source'])
    if '_u0_lgbm = _lgb_mod.train(' in src and 'UNKNOWN0_MLP_REPORT' in src:
        cell19_idx = i
        break
if cell19_idx is None:
    raise RuntimeError("Could not find Cell 19 (LightGBM training)")
print(f"Cell 19 found at index {cell19_idx}")

src19 = cell_src(cell19_idx)

# ── Replacement 1: cat overrides (remove turn_action_count) ──────────────────
OLD_CAT_OVERRIDES = '''\
UNKNOWN0_LGBM_CAT_OVERRIDES = [
    'card_id', 'my_active_id', 'op_active_id',
    'attack_id', 'target_card_id', 'stadium_id',
]
UNKNOWN0_LGBM_ALL_CAT = list(dict.fromkeys(
    UNKNOWN0_LGBM_CAT_OVERRIDES + UNKNOWN0_CATEGORICAL_FEATURES))'''

NEW_CAT_OVERRIDES = '''\
UNKNOWN0_LGBM_CAT_OVERRIDES = [
    'card_id', 'my_active_id', 'op_active_id',
    'attack_id', 'target_card_id', 'stadium_id',
]
UNKNOWN0_LGBM_ALL_CAT = list(dict.fromkeys(
    UNKNOWN0_LGBM_CAT_OVERRIDES + UNKNOWN0_CATEGORICAL_FEATURES))
# Exclude features that can't be reliably computed in inference
UNKNOWN0_LGBM_EXCLUDE_FEATURES = ['turn_action_count']'''

assert OLD_CAT_OVERRIDES in src19, "OLD_CAT_OVERRIDES not found!"
src19 = src19.replace(OLD_CAT_OVERRIDES, NEW_CAT_OVERRIDES)

# ── Replacement 2: dataset build → add group for ranking ─────────────────────
OLD_PREP = '''\
                _lgbm_prep = {
                    'feature_names': UNKNOWN0_NUMERIC_FEATURES + UNKNOWN0_CATEGORICAL_FEATURES,
                    'cat_features': UNKNOWN0_LGBM_ALL_CAT,
                    'cat_maps': _cat_maps,
                }

                def _encode_for_lgbm(df, prep):
                    feats = prep['feature_names']
                    cm = prep['cat_maps']
                    cats = set(prep['cat_features'])
                    out = {}
                    for c in feats:
                        if c in cm:
                            out[c] = df[c].fillna('UNK').astype(str).map(
                                cm[c]).fillna(0).astype('int32')
                        elif c in cats:
                            out[c] = df[c].fillna(0).astype('int32')
                        else:
                            out[c] = df[c].fillna(0).astype('float32')
                    return pd.DataFrame(out, index=df.index)

                X_tr = _encode_for_lgbm(train_df, _lgbm_prep)
                X_va = _encode_for_lgbm(valid_df, _lgbm_prep)
                y_tr = train_df['is_chosen'].astype(int)
                y_va = valid_df['is_chosen'].astype(int)

                # Sample weights from Delta-V / phase weighting
                if USE_MC_RETURN_WEIGHTS and 'mc_step_weight' in train_df.columns:
                    _sw = train_df['mc_step_weight'].to_numpy().astype(float)
                    _sw_mean = float(_sw.mean())
                    _w_tr = (_sw / max(_sw_mean, 1e-8)).astype(float) if _sw_mean > 1e-8 else None
                else:
                    _w_tr = None

                _cat_cols = [c for c in UNKNOWN0_LGBM_ALL_CAT if c in X_tr.columns]
                _dtr = _lgb_mod.Dataset(
                    X_tr, label=y_tr, weight=_w_tr,
                    categorical_feature=_cat_cols, free_raw_data=False)
                _dva = _lgb_mod.Dataset(
                    X_va, label=y_va, reference=_dtr,
                    categorical_feature=_cat_cols)'''

NEW_PREP = '''\
                _all_feats = [f for f in UNKNOWN0_NUMERIC_FEATURES + UNKNOWN0_CATEGORICAL_FEATURES
                              if f not in UNKNOWN0_LGBM_EXCLUDE_FEATURES]
                _lgbm_prep = {
                    'feature_names': _all_feats,
                    'cat_features': UNKNOWN0_LGBM_ALL_CAT,
                    'cat_maps': _cat_maps,
                    'exclude_features': UNKNOWN0_LGBM_EXCLUDE_FEATURES,
                }

                def _encode_for_lgbm(df, prep):
                    feats = prep['feature_names']
                    cm = prep['cat_maps']
                    cats = set(prep['cat_features'])
                    out = {}
                    for c in feats:
                        if c in cm:
                            out[c] = df[c].fillna('UNK').astype(str).map(
                                cm[c]).fillna(0).astype('int32')
                        elif c in cats:
                            out[c] = df[c].fillna(0).astype('int32')
                        else:
                            out[c] = df[c].fillna(0).astype('float32')
                    return pd.DataFrame(out, index=df.index)

                X_tr = _encode_for_lgbm(train_df, _lgbm_prep)
                X_va = _encode_for_lgbm(valid_df, _lgbm_prep)
                y_tr = train_df['is_chosen'].astype(int)
                y_va = valid_df['is_chosen'].astype(int)

                # Group sizes for LambdaRank: number of options per decision
                def _make_groups(df):
                    return df.groupby('decision_id', sort=False).size().values

                _g_tr = _make_groups(train_df)
                _g_va = _make_groups(valid_df)
                print(f'  LambdaRank groups: train={len(_g_tr)} decisions, valid={len(_g_va)} decisions')

                _cat_cols = [c for c in UNKNOWN0_LGBM_ALL_CAT if c in X_tr.columns]
                _dtr = _lgb_mod.Dataset(
                    X_tr, label=y_tr, group=_g_tr,
                    categorical_feature=_cat_cols, free_raw_data=False)
                _dva = _lgb_mod.Dataset(
                    X_va, label=y_va, group=_g_va, reference=_dtr,
                    categorical_feature=_cat_cols)'''

assert OLD_PREP in src19, "OLD_PREP not found in Cell 19!"
src19 = src19.replace(OLD_PREP, NEW_PREP)

# ── Replacement 3: LightGBM params → lambdarank ───────────────────────────────
OLD_PARAMS = '''\
                _lgbm_params = {
                    'objective': 'binary',
                    'metric': 'binary_logloss',
                    'learning_rate': 0.05,
                    'num_leaves': 63,
                    'min_child_samples': 20,
                    'feature_fraction': 0.8,
                    'bagging_fraction': 0.8,
                    'bagging_freq': 5,
                    'scale_pos_weight': 8.5,
                    'verbosity': -1,
                    'n_jobs': -1,
                }'''

NEW_PARAMS = '''\
                _lgbm_params = {
                    'objective': 'lambdarank',
                    'metric': 'ndcg',
                    'ndcg_eval_at': [1, 3],
                    'learning_rate': 0.05,
                    'num_leaves': 127,
                    'min_child_samples': 10,
                    'feature_fraction': 0.85,
                    'bagging_fraction': 0.85,
                    'bagging_freq': 5,
                    'verbosity': -1,
                    'n_jobs': -1,
                }'''

assert OLD_PARAMS in src19, "OLD_PARAMS not found in Cell 19!"
src19 = src19.replace(OLD_PARAMS, NEW_PARAMS)

# ── Replacement 4: AUC/logloss metrics → NDCG metrics ────────────────────────
OLD_METRICS = '''\
                try:
                    from sklearn.metrics import roc_auc_score, log_loss as _sk_logloss
                    _yva_arr = y_va.to_numpy() if hasattr(y_va, 'to_numpy') else y_va
                    row_auc = float(roc_auc_score(_yva_arr, _lgbm_pred)) if len(set(_yva_arr.tolist())) == 2 else None
                    row_logloss = float(_sk_logloss(_yva_arr, _lgbm_pred, labels=[0, 1]))
                except Exception:
                    row_auc = None
                    row_logloss = None'''

NEW_METRICS = '''\
                try:
                    from sklearn.metrics import roc_auc_score
                    _yva_arr = y_va.to_numpy() if hasattr(y_va, 'to_numpy') else y_va
                    row_auc = float(roc_auc_score(_yva_arr, _lgbm_pred)) if len(set(_yva_arr.tolist())) == 2 else None
                    # NDCG@1 = decision-level top1 accuracy under the ranking model
                    row_logloss = None  # not applicable for ranking
                except Exception:
                    row_auc = None
                    row_logloss = None'''

assert OLD_METRICS in src19, "OLD_METRICS not found in Cell 19!"
src19 = src19.replace(OLD_METRICS, NEW_METRICS)

# ── Replacement 5: LGBM report → remove scale_pos_weight, add ranking fields ──
OLD_REPORT = '''\
                UNKNOWN0_MLP_REPORT = {
                    'status': 'ok',
                    'model': 'lightgbm',
                    'rows': int(len(work)),
                    'train_rows': int(len(train_df)),
                    'valid_rows': int(len(valid_df)),
                    'train_decisions': int(train_df['decision_id'].nunique()),
                    'valid_decisions': int(valid_df['decision_id'].nunique()),
                    'best_iteration': int(_u0_lgbm.best_iteration),
                    'num_leaves': int(_lgbm_params['num_leaves']),
                    'row_auc': row_auc,
                    'row_logloss': row_logloss,'''

NEW_REPORT = '''\
                UNKNOWN0_MLP_REPORT = {
                    'status': 'ok',
                    'model': 'lightgbm_lambdarank',
                    'rows': int(len(work)),
                    'train_rows': int(len(train_df)),
                    'valid_rows': int(len(valid_df)),
                    'train_decisions': int(train_df['decision_id'].nunique()),
                    'valid_decisions': int(valid_df['decision_id'].nunique()),
                    'best_iteration': int(_u0_lgbm.best_iteration),
                    'num_leaves': int(_lgbm_params['num_leaves']),
                    'row_auc': row_auc,
                    'row_logloss': row_logloss,'''

assert OLD_REPORT in src19, "OLD_REPORT not found in Cell 19!"
src19 = src19.replace(OLD_REPORT, NEW_REPORT)

# ── Replacement 6: update inference code (exclude turn_action_count) ──────────
# In the _LGBM_INJ_CODE, the feature encoding loop builds X from feature_names.
# Since 'turn_action_count' is excluded from feature_names in prep, it's
# automatically excluded from X. The rows dict still has it as 0.0 (harmless).
# No code change needed in inference - the prep's feature_names controls this.

set_cell_src(cell19_idx, src19)

# Quick sanity checks
c19 = cell_src(cell19_idx)
assert 'lambdarank' in c19, "lambdarank not in Cell 19!"
assert 'turn_action_count' not in c19 or 'EXCLUDE_FEATURES' in c19, "turn_action_count not excluded!"
assert 'ndcg' in c19, "ndcg metric not in Cell 19!"
assert '_make_groups' in c19, "group building not in Cell 19!"
assert 'num_leaves\': 127' in c19, "num_leaves not updated!"
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

# ── Write notebook ─────────────────────────────────────────────────────────────
with open(DST_NB, 'w') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print(f"\nv05d19 notebook written: {DST_NB}")
print(f"Total cells: {len(nb['cells'])}")
