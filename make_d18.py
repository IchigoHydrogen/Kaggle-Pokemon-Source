"""Create v05d18 from v05d17: replace Unknown0 MLP (torch) with LightGBM.

Key change: card_id / my_active_id / op_active_id / attack_id / target_card_id / stadium_id
are now declared as CATEGORICAL in LightGBM. This lets the model learn
"card X → choose Y" rules directly, without assuming numeric ordering on IDs.
"""
import json, uuid
from pathlib import Path

WORK_DIR = Path('/kaggle/working')
SRC_NB = WORK_DIR / 'pokemon-20260625-v0-05d17.ipynb'
DST_NB = WORK_DIR / 'pokemon-20260625-v0-05d18.ipynb'

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
src1 = src1.replace("EXPERIMENT_NAME = 'v0_05d17_adv_top200_mlp'",
                    "EXPERIMENT_NAME = 'v0_05d18_lgbm_cat_id'")
src1 = src1.replace("RUN_PREFIX = 'pokemon-20260625-v0-05d17'",
                    "RUN_PREFIX = 'pokemon-20260625-v0-05d18'")
set_cell_src(1, src1)
assert 'v0_05d18_lgbm_cat_id' in cell_src(1)
assert 'pokemon-20260625-v0-05d18' in cell_src(1)
print("Cell 1: OK")

# ── Find Cell 19 (Unknown0 MLP training) ─────────────────────────────────────
cell19_idx = None
for i, cell in enumerate(nb['cells']):
    src = ''.join(cell['source'])
    if 'Unknown0OptionMLP' in src and 'UNKNOWN0_MLP_REPORT' in src:
        cell19_idx = i
        break
if cell19_idx is None:
    raise RuntimeError("Could not find Cell 19 (Unknown0 training)")
print(f"Cell 19 found at index {cell19_idx}")

src19 = cell_src(cell19_idx)

# ── Replacement 1: torch/MLP block → LightGBM block ─────────────────────────
# Replace from the torch import through the final print of UNKNOWN0_POLICY_SUMMARY.
# The new block produces the SAME output variables so all downstream code is unchanged.

OLD_MLP_BLOCK_START = (
    "try:\n"
    "    import torch\n"
    "    from torch import nn\n"
    "    TORCH_AVAILABLE_FOR_UNKNOWN0 = True\n"
    "except Exception as exc:\n"
    "    TORCH_AVAILABLE_FOR_UNKNOWN0 = False\n"
    "    UNKNOWN0_TORCH_IMPORT_ERROR = repr(exc)"
)

OLD_MLP_BLOCK_END = (
    "print('UNKNOWN0_MLP_REPORT status:', UNKNOWN0_MLP_REPORT.get('status'))\n"
    "print('UNKNOWN0_POLICY_SUMMARY:', UNKNOWN0_POLICY_SUMMARY)"
)

# Build the new LightGBM block as a multi-line string
NEW_LGBM_BLOCK = '''\
# v05d18: LightGBM with card IDs as categorical features
UNKNOWN0_LGBM_CAT_OVERRIDES = [
    'card_id', 'my_active_id', 'op_active_id',
    'attack_id', 'target_card_id', 'stadium_id',
]
UNKNOWN0_LGBM_ALL_CAT = list(dict.fromkeys(
    UNKNOWN0_LGBM_CAT_OVERRIDES + UNKNOWN0_CATEGORICAL_FEATURES))

try:
    import lightgbm as _lgb_mod
    _LGBM_OK = True
except Exception as _lgbm_import_err:
    _LGBM_OK = False
    print(f'LightGBM import failed: {_lgbm_import_err}')

if not _LGBM_OK:
    UNKNOWN0_MLP_REPORT = {'status': 'skipped', 'reason': 'lgbm_unavailable'}
elif 'ALAKAZAM_OPTION_MODEL_DF' not in globals() or ALAKAZAM_OPTION_MODEL_DF.empty:
    UNKNOWN0_MLP_REPORT = {'status': 'skipped', 'reason': 'empty_option_model_df'}
else:
    try:
        max_rows = int(os.environ.get('V05_UNKNOWN0_MLP_MAX_ROWS',
                       str(min(350_000, max(50_000, MLP_MAX_ROWS // 2)))))
        work = prepare_unknown0_training_frame(ALAKAZAM_OPTION_MODEL_DF, max_rows=max_rows)
        if work.empty or work['is_chosen'].nunique() < 2 or work['decision_id'].nunique() < 50:
            UNKNOWN0_MLP_REPORT = {'status': 'skipped', 'reason': 'insufficient_unknown0_rows',
                                    'rows': int(len(work))}
        else:
            train_df, valid_df = split_by_decision_safe(work, valid_frac=0.15)
            if train_df.empty or valid_df.empty or train_df['is_chosen'].nunique() < 2:
                UNKNOWN0_MLP_REPORT = {'status': 'skipped', 'reason': 'bad_unknown0_split'}
            else:
                # Build categorical encoding maps (string→int only for string columns;
                # integer card-IDs are passed directly as ints to LightGBM categorical)
                _cat_maps = {}
                for _col in UNKNOWN0_LGBM_ALL_CAT:
                    if _col in train_df.columns and train_df[_col].dtype == object:
                        _uniq = sorted(set(
                            train_df[_col].fillna('UNK').astype(str).unique().tolist() +
                            valid_df[_col].fillna('UNK').astype(str).unique().tolist()))
                        if 'UNK' not in _uniq:
                            _uniq = ['UNK'] + _uniq
                        else:
                            _uniq = ['UNK'] + [v for v in _uniq if v != 'UNK']
                        _cat_maps[_col] = {v: i for i, v in enumerate(_uniq)}

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
                    categorical_feature=_cat_cols)

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
                }
                _u0_lgbm = _lgb_mod.train(
                    _lgbm_params, _dtr,
                    num_boost_round=500,
                    valid_sets=[_dtr, _dva],
                    valid_names=['train', 'valid'],
                    callbacks=[
                        _lgb_mod.early_stopping(30, verbose=False),
                        _lgb_mod.log_evaluation(50),
                    ],
                )
                print(f'unknown0 lgbm: best_iter={_u0_lgbm.best_iteration}')

                # Predict on validation set
                _lgbm_pred = _u0_lgbm.predict(X_va).astype(float)
                pred_df = valid_df.copy()
                pred_df['unknown0_mlp_pred'] = _lgbm_pred  # keep col name for downstream compat
                pred_df['rule_proxy_score'] = offline_rule_proxy_scores(pred_df)
                pred_df['bc_pred'] = predict_with_bc_models(pred_df) if 'predict_with_bc_models' in globals() else np.nan
                UNKNOWN0_MLP_VALID_PRED_DF = pred_df

                UNKNOWN0_POLICY_FIT_PRED_DF, UNKNOWN0_POLICY_FIT_SPLIT_SUMMARY = filter_unknown0_policy_fit_frame(
                    pred_df, EPISODE_SPLIT_DF, excluded_splits=('holdout',))

                # Early-game filter (same as MLP path)
                _egf = globals().get('EARLY_GAME_ONLY_POLICY_FIT', False)
                if _egf and not UNKNOWN0_POLICY_FIT_PRED_DF.empty:
                    _ep_ns = EPISODE_SPLIT_DF[['episode_id','n_steps']].drop_duplicates('episode_id') if 'n_steps' in EPISODE_SPLIT_DF.columns else None
                    _fc = UNKNOWN0_POLICY_FIT_PRED_DF.copy()
                    if _ep_ns is not None and not _ep_ns.empty and 'episode_id' in _fc.columns and 'step' in _fc.columns:
                        _fc['episode_id'] = _fc['episode_id'].astype(str)
                        _ep_ns = _ep_ns.copy(); _ep_ns['episode_id'] = _ep_ns['episode_id'].astype(str)
                        _fc = _fc.merge(_ep_ns, on='episode_id', how='left')
                        _fc['_step_frac'] = _fc['step'].astype(float) / _fc['n_steps'].astype(float).clip(lower=1)
                        _early = _fc[_fc['_step_frac'] <= 2/3].drop(columns=['_step_frac','n_steps'], errors='ignore')
                        if len(_early) >= 20:
                            print(f'[early_game_filter] kept {len(_early)}/{len(UNKNOWN0_POLICY_FIT_PRED_DF)} rows (step_frac<=2/3)')
                            UNKNOWN0_POLICY_FIT_PRED_DF = _early
                        else:
                            print('[early_game_filter] too few early rows, using full dataset')
                    else:
                        print('[early_game_filter] missing step/n_steps columns, skipping filter')

                UNKNOWN0_MLP_SIGNATURE_REPORT_DF = build_unknown0_signature_report(
                    pred_df, UNKNOWN0_MLP_MARGIN_THRESHOLD)
                UNKNOWN0_POLICY_FIT_SIGNATURE_REPORT_DF = build_unknown0_signature_report(
                    UNKNOWN0_POLICY_FIT_PRED_DF, UNKNOWN0_MLP_MARGIN_THRESHOLD)
                allowed = select_unknown0_policy_signatures(UNKNOWN0_POLICY_FIT_SIGNATURE_REPORT_DF)
                UNKNOWN0_MLP_CALIBRATION_DF = build_unknown0_calibration(pred_df, 'unknown0_mlp_pred')
                UNKNOWN0_RULE_PROXY_ASSIST_REPORT_DF = build_unknown0_assist_report(
                    pred_df, allowed, UNKNOWN0_MLP_MARGIN_THRESHOLD)
                UNKNOWN0_POLICY_TABLE, UNKNOWN0_POLICY_TABLE_DF, UNKNOWN0_POLICY_SUMMARY = \
                    build_unknown0_policy_table(UNKNOWN0_POLICY_FIT_PRED_DF, allowed,
                                                UNKNOWN0_MLP_MARGIN_THRESHOLD)
                UNKNOWN0_POLICY_SUMMARY['fit_split_filter'] = UNKNOWN0_POLICY_FIT_SPLIT_SUMMARY

                try:
                    from sklearn.metrics import roc_auc_score, log_loss as _sk_logloss
                    _yva_arr = y_va.to_numpy() if hasattr(y_va, 'to_numpy') else y_va
                    row_auc = float(roc_auc_score(_yva_arr, _lgbm_pred)) if len(set(_yva_arr.tolist())) == 2 else None
                    row_logloss = float(_sk_logloss(_yva_arr, _lgbm_pred, labels=[0, 1]))
                except Exception:
                    row_auc = None
                    row_logloss = None

                # Save LightGBM model + prep
                _lgbm_txt_path = MODEL_DIR / f'{RUN_PREFIX}-unknown0_lgbm_reranker.txt'
                _lgbm_prep_path = MODEL_DIR / f'{RUN_PREFIX}-unknown0_lgbm_prep.json'
                _u0_lgbm.save_model(str(_lgbm_txt_path))
                with open(_lgbm_prep_path, 'w') as _jf:
                    json.dump(_lgbm_prep, _jf, ensure_ascii=False)

                UNKNOWN0_DIRECT_MLP_PACKAGING_STATUS = {
                    'status': 'ok',
                    'model': str(_lgbm_txt_path),
                    'prep': str(_lgbm_prep_path),
                    'best_iteration': int(_u0_lgbm.best_iteration),
                }

                ARTIFACT_PATHS['unknown0_lgbm_model'] = str(_lgbm_txt_path)
                ARTIFACT_PATHS['unknown0_lgbm_prep'] = str(_lgbm_prep_path)
                ARTIFACT_PATHS['unknown0_mlp_valid_predictions'] = safe_save_table(
                    UNKNOWN0_MLP_VALID_PRED_DF, OUTPUT_DIR / 'unknown0_mlp_valid_predictions')
                ARTIFACT_PATHS['unknown0_mlp_signature_report'] = safe_save_table(
                    UNKNOWN0_MLP_SIGNATURE_REPORT_DF, OUTPUT_DIR / 'unknown0_mlp_signature_report')
                ARTIFACT_PATHS['unknown0_mlp_calibration'] = safe_save_table(
                    UNKNOWN0_MLP_CALIBRATION_DF, OUTPUT_DIR / 'unknown0_mlp_calibration')
                ARTIFACT_PATHS['unknown0_policy_fit_signature_report'] = safe_save_table(
                    UNKNOWN0_POLICY_FIT_SIGNATURE_REPORT_DF, OUTPUT_DIR / 'unknown0_policy_fit_signature_report')
                ARTIFACT_PATHS['unknown0_rule_proxy_assist_report'] = safe_save_table(
                    UNKNOWN0_RULE_PROXY_ASSIST_REPORT_DF, OUTPUT_DIR / 'unknown0_rule_proxy_assist_report')
                ARTIFACT_PATHS['unknown0_policy_table'] = safe_save_table(
                    UNKNOWN0_POLICY_TABLE_DF, OUTPUT_DIR / 'unknown0_policy_table')
                ARTIFACT_PATHS['unknown0_policy_table_json'] = write_json(
                    OUTPUT_DIR / 'unknown0_policy_table.json', UNKNOWN0_POLICY_TABLE)

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
                    'row_logloss': row_logloss,
                    'unknown0_lgbm_decision_metrics': grouped_prediction_metrics(pred_df, 'unknown0_mlp_pred'),
                    'rule_proxy_decision_metrics': grouped_prediction_metrics(pred_df, 'rule_proxy_score'),
                    'bc_decision_metrics': grouped_prediction_metrics(
                        pred_df[pred_df['bc_pred'].notna()].copy(), 'bc_pred')
                        if 'bc_pred' in pred_df.columns and pred_df['bc_pred'].notna().any() else {'decisions': 0},
                    'feature_safety': UNKNOWN0_FEATURE_SAFETY,
                    'allowed_signatures': sorted(allowed),
                    'policy_table_entries': int(len(UNKNOWN0_POLICY_TABLE)),
                    'policy_fit_split_summary': UNKNOWN0_POLICY_FIT_SPLIT_SUMMARY,
                    'direct_mlp_packaging_status': UNKNOWN0_DIRECT_MLP_PACKAGING_STATUS,
                    'feature_importance_top10': dict(sorted(
                        zip(_u0_lgbm.feature_name(),
                            _u0_lgbm.feature_importance(importance_type='gain').tolist()),
                        key=lambda x: -x[1])[:10]),
                }
                print('unknown0 lgbm report:', UNKNOWN0_MLP_REPORT)
                if not UNKNOWN0_MLP_SIGNATURE_REPORT_DF.empty:
                    display(UNKNOWN0_MLP_SIGNATURE_REPORT_DF.head(20)) if 'display' in globals() else print(UNKNOWN0_MLP_SIGNATURE_REPORT_DF.head(20))
                if not UNKNOWN0_RULE_PROXY_ASSIST_REPORT_DF.empty:
                    display(UNKNOWN0_RULE_PROXY_ASSIST_REPORT_DF) if 'display' in globals() else print(UNKNOWN0_RULE_PROXY_ASSIST_REPORT_DF)
    except Exception as exc:
        UNKNOWN0_MLP_REPORT = {'status': 'error', 'error': repr(exc), 'traceback': traceback.format_exc(limit=8)}
        log_error('unknown0_lgbm', error=repr(exc), traceback=traceback.format_exc(limit=8))

ARTIFACT_PATHS['unknown0_mlp_report'] = write_json(OUTPUT_DIR / 'unknown0_mlp_report.json', UNKNOWN0_MLP_REPORT)
ARTIFACT_PATHS['unknown0_policy_summary'] = write_json(OUTPUT_DIR / 'unknown0_policy_summary.json', UNKNOWN0_POLICY_SUMMARY)
ARTIFACT_PATHS['unknown0_policy_fit_split_summary'] = write_json(OUTPUT_DIR / 'unknown0_policy_fit_split_summary.json', UNKNOWN0_POLICY_FIT_SPLIT_SUMMARY)
ARTIFACT_PATHS['unknown0_direct_mlp_packaging_status'] = write_json(OUTPUT_DIR / 'unknown0_direct_mlp_packaging_status.json', UNKNOWN0_DIRECT_MLP_PACKAGING_STATUS)
print('UNKNOWN0_MLP_REPORT status:', UNKNOWN0_MLP_REPORT.get('status'))
print('UNKNOWN0_POLICY_SUMMARY:', UNKNOWN0_POLICY_SUMMARY)'''

# Find exact old block in src19
idx_start = src19.find(OLD_MLP_BLOCK_START)
if idx_start < 0:
    raise RuntimeError("OLD_MLP_BLOCK_START not found in Cell 19!")
idx_end = src19.find(OLD_MLP_BLOCK_END, idx_start)
if idx_end < 0:
    raise RuntimeError("OLD_MLP_BLOCK_END not found in Cell 19!")
idx_end += len(OLD_MLP_BLOCK_END)

old_block = src19[idx_start:idx_end]
print(f"Replacing MLP block: pos {idx_start}-{idx_end} ({idx_end-idx_start} chars)")
src19 = src19[:idx_start] + NEW_LGBM_BLOCK + src19[idx_end:]
print("Cell 19: MLP → LightGBM block replaced")

# ── Replacement 2: _MLP_INJ_CODE → _LGBM_INJ_CODE ────────────────────────────
LGBM_INJ_CODE = r'''
# v0-05d18: UNKNOWN_0 LightGBM inference. Falls back to policy table on failure.

_U0_LGBM = None

def _u0_load_lgbm():
    global _U0_LGBM
    try:
        import os as _os, lightgbm as _lgb, json as _json
        _d = _os.path.dirname(_os.path.abspath(__file__))
        _mp = _os.path.join(_d, 'unknown0_lgbm.txt')
        _pp = _os.path.join(_d, 'unknown0_lgbm_prep.json')
        if not _os.path.exists(_mp) or not _os.path.exists(_pp):
            return
        _model = _lgb.Booster(model_file=_mp)
        with open(_pp) as _f:
            _prep = _json.load(_f)
        _U0_LGBM = {'model': _model, 'prep': _prep}
    except Exception:
        pass

_u0_load_lgbm()


def _u0_lgbm_scores(obs, select, my_index, pi_opp):
    """Return per-option LightGBM scores for UNKNOWN_0, or None on failure."""
    if _U0_LGBM is None:
        return None
    try:
        import numpy as _np
        my_ps = obs.current.players[my_index]
        op_ps = obs.current.players[pi_opp]
        my_active = (my_ps.active or [None])[0]
        op_active = (op_ps.active or [None])[0]
        my_bench = [p for p in (my_ps.bench or []) if p is not None]
        op_bench = [p for p in (op_ps.bench or []) if p is not None]
        my_all = ([my_active] if my_active else []) + my_bench
        my_ids = [getattr(p, 'id', 0) for p in my_all if p is not None]
        my_hand_n = float(len(my_ps.hand or []))
        my_deck_n = float(getattr(my_ps, 'deckCount', len(getattr(my_ps, 'deck', []) or [])))
        op_hp = float(getattr(op_active, 'hp', 0) or 0)
        powerful_hand = my_hand_n * 20.0
        sig = _unknown0_policy_abstract_sig(select) if 'USE_ABSTRACT_OPTION_SIGNATURE' in globals() and USE_ABSTRACT_OPTION_SIGNATURE else 'N0'
        rows = []
        for i, o in enumerate(select.option):
            ot = getattr(o, 'type', None)
            ot_n = (str(getattr(ot, 'name', '') or '').upper() or
                    str(ot).split('.')[-1].upper() or 'UNK')
            rows.append({
                'option_index': float(i),
                'num_options': float(len(select.option)),
                'min_count': float(getattr(select, 'minCount', 1) or 1),
                'max_count': float(getattr(select, 'maxCount', 1) or 1),
                'card_id': float(getattr(o, 'id', 0) or 0),
                'target_card_id': float(getattr(o, 'targetId', 0) or 0),
                'attack_id': float(getattr(o, 'attackId', 0) or 0),
                'number_value': float(getattr(o, 'number', 0) or 0),
                'in_play_index': float(getattr(o, 'inPlayIndex', 0) or 0),
                'remain_damage_counter': float(getattr(o, 'remainDamageCounter', 0) or 0),
                'remain_energy_cost': float(getattr(o, 'remainEnergyCost', 0) or 0),
                'turn': float(obs.current.turn),
                'turn_action_count': 0.0,
                'my_active_id': float(getattr(my_active, 'id', 0) or 0),
                'my_active_hp': float(getattr(my_active, 'hp', 0) or 0),
                'my_active_energy_count': float(len(getattr(my_active, 'energies', []) or [])),
                'op_active_id': float(getattr(op_active, 'id', 0) or 0),
                'op_active_hp': op_hp,
                'op_active_energy_count': float(len(getattr(op_active, 'energies', []) or [])),
                'my_bench_count': float(len(my_bench)),
                'op_bench_count': float(len(op_bench)),
                'my_alakazam_count': float(sum(1 for _id in my_ids if _id == 743)),
                'my_kadabra_count': float(sum(1 for _id in my_ids if _id == 742)),
                'my_abra_count': float(sum(1 for _id in my_ids if _id == 741)),
                'my_dudunsparce_count': float(sum(1 for _id in my_ids if _id == 66)),
                'my_hand_count': my_hand_n,
                'op_hand_count': float(len(op_ps.hand or [])),
                'my_deck_count': my_deck_n,
                'op_deck_count': float(getattr(op_ps, 'deckCount', 0)),
                'my_prizes_left': float(len([p for p in (my_ps.prize or []) if p is not None])),
                'op_prizes_left': float(len([p for p in (op_ps.prize or []) if p is not None])),
                'stadium_id': float(getattr(getattr(obs.current, 'stadium', None), 'id', 0) or 0),
                'powerful_hand_damage_est': powerful_hand,
                'powerful_hand_can_ko_active': float(powerful_hand >= op_hp),
                'deckout_risk_feature': float(my_deck_n <= 4),
                'context_name': 'UNKNOWN_0',
                'option_type': ot_n,
                'area': str(getattr(o, 'area', 0) or 0),
                'in_play_area': str(getattr(o, 'inPlayArea', 0) or 0),
                'option_signature': sig,
            })
        prep = _U0_LGBM['prep']
        feature_names = prep['feature_names']
        cat_maps = prep.get('cat_maps', {})
        cat_set = set(prep.get('cat_features', []))
        X = _np.zeros((len(rows), len(feature_names)), dtype='float64')
        for ci, col in enumerate(feature_names):
            if col in cat_maps:
                m = cat_maps[col]
                for ri, r in enumerate(rows):
                    X[ri, ci] = float(m.get(str(r.get(col, 'UNK') or 'UNK'), 0))
            elif col in cat_set:
                for ri, r in enumerate(rows):
                    X[ri, ci] = float(r.get(col, 0) or 0)
            else:
                for ri, r in enumerate(rows):
                    X[ri, ci] = float(r.get(col, 0) or 0)
        return _U0_LGBM['model'].predict(X).tolist()
    except Exception:
        return None
'''

# The call-site patch for main.py
LGBM_CALL_PATCH_TARGET = '    unknown0_policy_selected = _unknown0_policy_select('
LGBM_CALL_PATCH_REPL = (
    '    if _U0_LGBM is not None and _unknown0_policy_is_context(context):\n'
    '        _lgbm_scores = _u0_lgbm_scores(obs, select, my_index, 1 - my_index)\n'
    '        if _lgbm_scores and len(_lgbm_scores) == len(select.option):\n'
    '            _best_i = max(range(len(_lgbm_scores)), key=lambda _i: _lgbm_scores[_i])\n'
    '            _sel = safe_unique_action([_best_i], len(select.option), min_count, max_count)\n'
    '            if len(_sel) >= min_count:\n'
    '                return _sel\n'
    '    unknown0_policy_selected = _unknown0_policy_select('
)

# Find and replace _MLP_INJ_CODE section in src19
OLD_INJ_CODE_MARKER = "# ── v05d16: UNKNOWN_0 MLP direct submission source builder ──────────────────\n_MLP_INJ_CODE = "
NEW_INJ_CODE_SECTION = (
    "# ── v05d18: UNKNOWN_0 LightGBM direct submission source builder ──────────────────\n"
    "_LGBM_INJ_CODE = " + repr(LGBM_INJ_CODE) + "\n"
    "\n"
    "_LGBM_PATCH_TARGET = '    unknown0_policy_selected = _unknown0_policy_select('\n"
    "_LGBM_PATCH_REPL = (\n"
    "    '    if _U0_LGBM is not None and _unknown0_policy_is_context(context):\\n'\n"
    "    '        _lgbm_scores = _u0_lgbm_scores(obs, select, my_index, 1 - my_index)\\n'\n"
    "    '        if _lgbm_scores and len(_lgbm_scores) == len(select.option):\\n'\n"
    "    '            _best_i = max(range(len(_lgbm_scores)), key=lambda _i: _lgbm_scores[_i])\\n'\n"
    "    '            _sel = safe_unique_action([_best_i], len(select.option), min_count, max_count)\\n'\n"
    "    '            if len(_sel) >= min_count:\\n'\n"
    "    '                return _sel\\n'\n"
    "    '    unknown0_policy_selected = _unknown0_policy_select('\n"
    ")\n"
    "\n"
    "def make_unknown0_lgbm_submission_agent_source(policy_table_source):\n"
    "    source = policy_table_source\n"
    "    marker = 'def agent(obs_dict: dict) -> list[int]:'\n"
    "    if marker not in source:\n"
    "        return source\n"
    "    idx = source.find(marker)\n"
    "    source = source[:idx] + _LGBM_INJ_CODE + '\\n\\n' + source[idx:]\n"
    "    if _LGBM_PATCH_TARGET in source:\n"
    "        source = source.replace(_LGBM_PATCH_TARGET, _LGBM_PATCH_REPL)\n"
    "    return source\n"
    "\n"
    "print('make_unknown0_lgbm_submission_agent_source: defined')\n"
)

# Find old MLP injection section - from the builder comment to end of the function
OLD_INJ_START = src19.find(OLD_INJ_CODE_MARKER)
if OLD_INJ_START < 0:
    raise RuntimeError(f"OLD_INJ_CODE_MARKER not found! Looking for: {repr(OLD_INJ_CODE_MARKER[:50])}")

# Find end of file (the whole section from the marker to end)
old_tail = src19[OLD_INJ_START:]
src19 = src19[:OLD_INJ_START] + NEW_INJ_CODE_SECTION
print(f"Cell 19: injection code replaced ({len(old_tail)} chars removed, {len(NEW_INJ_CODE_SECTION)} chars added)")

set_cell_src(cell19_idx, src19)
assert 'lightgbm' in cell_src(cell19_idx).lower(), "LightGBM not in Cell 19!"
assert 'make_unknown0_lgbm_submission_agent_source' in cell_src(cell19_idx), "Builder function missing!"
assert 'Unknown0OptionMLP' not in cell_src(cell19_idx), "MLP class still present!"
print("Cell 19: OK")

# ── Cell 20: update submission building for LightGBM ─────────────────────────
cell20_idx = None
for i, cell in enumerate(nb['cells']):
    src = ''.join(cell['source'])
    if 'build_submission_archive' in src and 'SUBMISSION_VARIANT_SOURCES' in src and 'tarfile' in src:
        cell20_idx = i
        break
if cell20_idx is None:
    raise RuntimeError("Could not find Cell 20 (submission building)")
print(f"Cell 20 found at index {cell20_idx}")

src20 = cell_src(cell20_idx)

# Replace model file reference and builder function call
OLD_MLP_MODEL = (
    "_u0_model_pt = MODEL_DIR / f'{RUN_PREFIX}-unknown0_mlp_reranker.pt'\n"
    "if 'make_unknown0_mlp_direct_submission_agent_source' in globals() and _u0_model_pt.exists():\n"
    "    try:\n"
    "        UNKNOWN0_MLP_DIRECT_MAIN_SOURCE = make_unknown0_mlp_direct_submission_agent_source(\n"
    "            UNKNOWN0_POLICY_MAIN_SOURCE\n"
    "        )\n"
    "        UNKNOWN0_MLP_DIRECT_STATUS = {'status': 'ok', 'model': str(_u0_model_pt)}"
)

NEW_LGBM_MODEL = (
    "_u0_lgbm_txt = MODEL_DIR / f'{RUN_PREFIX}-unknown0_lgbm_reranker.txt'\n"
    "_u0_lgbm_prep = MODEL_DIR / f'{RUN_PREFIX}-unknown0_lgbm_prep.json'\n"
    "if 'make_unknown0_lgbm_submission_agent_source' in globals() and _u0_lgbm_txt.exists() and _u0_lgbm_prep.exists():\n"
    "    try:\n"
    "        UNKNOWN0_MLP_DIRECT_MAIN_SOURCE = make_unknown0_lgbm_submission_agent_source(\n"
    "            UNKNOWN0_POLICY_MAIN_SOURCE\n"
    "        )\n"
    "        UNKNOWN0_MLP_DIRECT_STATUS = {'status': 'ok', 'model': str(_u0_lgbm_txt)}"
)

if OLD_MLP_MODEL in src20:
    src20 = src20.replace(OLD_MLP_MODEL, NEW_LGBM_MODEL)
    print("Cell 20: model path patched")
else:
    print("WARNING: OLD_MLP_MODEL not found in Cell 20!")
    idx_dbg = src20.find("unknown0_mlp_reranker.pt")
    print(f"  'unknown0_mlp_reranker.pt' at: {idx_dbg}")

# Replace extra_files: .pt → .txt + _prep.json
OLD_EXTRA_FILES = (
    "    'unknown0_mlp_direct': ([(str(_u0_model_pt), 'unknown0_mlp.pt')]\n"
    "                            if _u0_model_pt.exists() else []),"
)
NEW_EXTRA_FILES = (
    "    'unknown0_mlp_direct': ([(str(_u0_lgbm_txt), 'unknown0_lgbm.txt'),\n"
    "                              (str(_u0_lgbm_prep), 'unknown0_lgbm_prep.json')]\n"
    "                             if _u0_lgbm_txt.exists() and _u0_lgbm_prep.exists() else []),"
)

if OLD_EXTRA_FILES in src20:
    src20 = src20.replace(OLD_EXTRA_FILES, NEW_EXTRA_FILES)
    print("Cell 20: extra_files patched")
else:
    print("WARNING: OLD_EXTRA_FILES not found in Cell 20!")

# Patch MLP-specific assertions → LGBM assertions
OLD_MLP_ASSERTS = (
    "        print(f'MLP direct source built, len={len(UNKNOWN0_MLP_DIRECT_MAIN_SOURCE)}')\n"
    "        assert '_U0_MLP_BUNDLE' in UNKNOWN0_MLP_DIRECT_MAIN_SOURCE, 'MLP code not injected'\n"
    "        assert '_u0_infer_scores' in UNKNOWN0_MLP_DIRECT_MAIN_SOURCE, 'infer function missing'"
)
NEW_LGBM_ASSERTS = (
    "        print(f'LGBM direct source built, len={len(UNKNOWN0_MLP_DIRECT_MAIN_SOURCE)}')\n"
    "        assert '_u0_load_lgbm' in UNKNOWN0_MLP_DIRECT_MAIN_SOURCE, 'LGBM code not injected'\n"
    "        assert '_u0_lgbm_scores' in UNKNOWN0_MLP_DIRECT_MAIN_SOURCE, 'LGBM score function missing'"
)
if OLD_MLP_ASSERTS in src20:
    src20 = src20.replace(OLD_MLP_ASSERTS, NEW_LGBM_ASSERTS)
    print("Cell 20: assertions patched")
else:
    print("WARNING: OLD_MLP_ASSERTS not found in Cell 20!")

# Patch except/else messages
OLD_EXCEPT_PRINT = "        print(f'MLP direct source build failed: {_exc}')"
NEW_EXCEPT_PRINT = "        print(f'LGBM direct source build failed: {_exc}')"
if OLD_EXCEPT_PRINT in src20:
    src20 = src20.replace(OLD_EXCEPT_PRINT, NEW_EXCEPT_PRINT)

OLD_ELSE_SKIP = (
    "    UNKNOWN0_MLP_DIRECT_STATUS = {'status': 'skipped',\n"
    "        'reason': f'model_missing={not _u0_model_pt.exists()}'}\n"
    "    print(f'MLP direct: skipped ({UNKNOWN0_MLP_DIRECT_STATUS[\"reason\"]})')"
)
NEW_ELSE_SKIP = (
    "    UNKNOWN0_MLP_DIRECT_STATUS = {'status': 'skipped',\n"
    "        'reason': f'model_missing={not _u0_lgbm_txt.exists()}'}\n"
    "    print(f'LGBM direct: skipped ({UNKNOWN0_MLP_DIRECT_STATUS[\"reason\"]})')"
)
if OLD_ELSE_SKIP in src20:
    src20 = src20.replace(OLD_ELSE_SKIP, NEW_ELSE_SKIP)
    print("Cell 20: else/except messages patched")
else:
    print("WARNING: OLD_ELSE_SKIP not found in Cell 20!")

set_cell_src(cell20_idx, src20)
assert '_u0_lgbm_txt' in cell_src(cell20_idx), "Cell 20 lgbm path missing!"
assert 'unknown0_lgbm.txt' in cell_src(cell20_idx), "Cell 20 lgbm.txt missing!"
assert '_U0_MLP_BUNDLE' not in cell_src(cell20_idx), "Cell 20 old MLP assertion still present!"
print("Cell 20: OK")

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
print(f"\nv05d18 notebook written: {DST_NB}")
print(f"Total cells: {len(nb['cells'])}")
