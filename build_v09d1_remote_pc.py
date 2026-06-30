"""Build pokemon-20260627-v0-09d1-remote-pc.ipynb from v08d34.

v09d1: INFERENCE-FAITHFUL EVALUATION (Track A foundation).

THE PROBLEM (discovered by reading main.py):
  At inference, several features collapse to constants/proxies:
    - position_winrate = 0.5 (constant)
    - steps_since_op   = 2.0 (constant)
    - op_last_context  = derived from op_active energyCount, NOT the real context:
        'UNKNOWN_0' if energy>=2 else ('ATTACH_FROM' if energy==1 else 'NONE')
  But the LOCAL holdout eval uses the REAL mined feature values.
  => train/serve skew is INVISIBLE to local top1. We cannot judge Track A fixes
     without spending Kaggle submissions (limited to 5/day).

THE FIX (this experiment):
  Add a second eval pass on the SAME holdout where the proxied features are
  recomputed EXACTLY as main.py does at inference. Report both:
    - real_top1     (current metric, uses mined real features)
    - infer_top1    (inference-faithful, uses the same proxies main.py uses)
    - skew          = real_top1 - infer_top1  (how much we lose to train/serve skew)
  Also diagnose how many holdout rows have op_last_context proxy != real.

  infer_top1 becomes the NEW primary optimization target — it is the
  locally-measurable proxy for Kaggle behavior.

No model change — v09d1 IS v08d34 (lambdarank+trunc1+position_winrate) plus the
extra eval pass. Same weights, same real_top1=0.5485 expected.

Based on v08d34 (best, Kaggle 792.1).

Changes from v08d34:
  1. EXPERIMENT_NAME → v0_09d1_infer_faithful
  2. RUN_PREFIX → pokemon-20260627-v0-09d1
  3. Cell[7]: Preload from v08d19 (same split)
  4. Cell[19]: add inference-faithful eval pass + skew diagnostics + report key
"""
import json

SRC = '/kaggle/working/pokemon-20260627-v0-08d34.ipynb'
DST = '/kaggle/working/pokemon-20260627-v0-09d1.ipynb'

with open(SRC) as f:
    nb = json.load(f)

cells = nb['cells']

# ── Cell[1]: Config ───────────────────────────────────────────────────────────
src1 = ''.join(cells[1]['source'])

OLD_EXP = "EXPERIMENT_NAME = 'v0_08d34_trunc1'"
NEW_EXP = "EXPERIMENT_NAME = 'v0_09d1_infer_faithful'"
assert OLD_EXP in src1
src1 = src1.replace(OLD_EXP, NEW_EXP)

OLD_PFX = "RUN_PREFIX = 'pokemon-20260627-v0-08d34'"
NEW_PFX = "RUN_PREFIX = 'pokemon-20260627-v0-09d1'"
assert OLD_PFX in src1
src1 = src1.replace(OLD_PFX, NEW_PFX)

cells[1]['source'] = src1.splitlines(keepends=True)

# ── Cell[7]: Preload ──────────────────────────────────────────────────────────
src7 = ''.join(cells[7]['source'])

OLD_PRELOAD = "# v08d34: preload from v08d19 (same split)\n"
NEW_PRELOAD = "# v09d1: preload from v08d19 (same split)\n"
assert OLD_PRELOAD in src7
src7 = src7.replace(OLD_PRELOAD, NEW_PRELOAD)

OLD_PRINT = "    print(f'v08d34 preload: {len(DECISION_ROWS_DF)} decisions from {_pfx}')\n"
NEW_PRINT = "    print(f'v09d1 preload: {len(DECISION_ROWS_DF)} decisions from {_pfx}')\n"
assert OLD_PRINT in src7
src7 = src7.replace(OLD_PRINT, NEW_PRINT)

OLD_FAIL = "    print('v08d34 preload: no cache found, mining from scratch')\n"
NEW_FAIL = "    print('v09d1 preload: no cache found, mining from scratch')\n"
assert OLD_FAIL in src7
src7 = src7.replace(OLD_FAIL, NEW_FAIL)

cells[7]['source'] = src7.splitlines(keepends=True)

# ── Cell[19]: add inference-faithful eval right after pred_df is built ─────────
src19 = ''.join(cells[19]['source'])

OLD_PRED = (
    "_lgbm_pred = _u0_lgbm.predict(X_va).astype(float)\n"
    "                pred_df = valid_df.copy()\n"
    "                pred_df['unknown0_mlp_pred'] = _lgbm_pred  # keep col name for downstream compat\n"
)
NEW_PRED = (
    "_lgbm_pred = _u0_lgbm.predict(X_va).astype(float)\n"
    "                pred_df = valid_df.copy()\n"
    "                pred_df['unknown0_mlp_pred'] = _lgbm_pred  # keep col name for downstream compat\n"
    "                # v09d1: INFERENCE-FAITHFUL eval — recompute proxied feats EXACTLY as main.py\n"
    "                UNKNOWN0_INFER_FAITHFUL_METRICS = {}\n"
    "                try:\n"
    "                    _vdf_if = valid_df.copy()\n"
    "                    _olc_found = 'op_last_context' in _vdf_if.columns and 'op_active_energy_count' in _vdf_if.columns\n"
    "                    if 'position_winrate' in _vdf_if.columns:\n"
    "                        _vdf_if['position_winrate'] = 0.5\n"
    "                    if 'steps_since_op' in _vdf_if.columns:\n"
    "                        _vdf_if['steps_since_op'] = 2.0\n"
    "                    if _olc_found:\n"
    "                        _oae = _vdf_if['op_active_energy_count'].fillna(0).astype(float)\n"
    "                        _vdf_if['op_last_context'] = _oae.map(\n"
    "                            lambda e: 'UNKNOWN_0' if e >= 2 else ('ATTACH_FROM' if e == 1 else 'NONE'))\n"
    "                    _X_va_if = _encode_for_lgbm(_vdf_if, _lgbm_prep)\n"
    "                    _pred_df_if = _vdf_if.copy()\n"
    "                    _pred_df_if['unknown0_mlp_pred'] = _u0_lgbm.predict(_X_va_if).astype(float)\n"
    "                    _m_real = grouped_prediction_metrics(pred_df, 'unknown0_mlp_pred')\n"
    "                    _m_if = grouped_prediction_metrics(_pred_df_if, 'unknown0_mlp_pred')\n"
    "                    UNKNOWN0_INFER_FAITHFUL_METRICS = _m_if\n"
    "                    _olc_diff = 0\n"
    "                    if _olc_found:\n"
    "                        _olc_diff = int((valid_df['op_last_context'].astype(str).values\n"
    "                                         != _vdf_if['op_last_context'].astype(str).values).sum())\n"
    "                    _rt = float(_m_real.get('top1', 0.0)); _it = float(_m_if.get('top1', 0.0))\n"
    "                    print(f'v09d1 INFER-FAITHFUL: real_top1={_rt:.4f} infer_top1={_it:.4f} '\n"
    "                          f'skew={_rt-_it:+.4f} | olc_found={_olc_found} '\n"
    "                          f'olc_proxy!=real={_olc_diff}/{len(valid_df)}')\n"
    "                except Exception as _e_if:\n"
    "                    print(f'v09d1 INFER-FAITHFUL eval failed: {_e_if!r}')\n"
)
assert OLD_PRED in src19, 'predict/pred_df anchor not found in Cell[19]'
src19 = src19.replace(OLD_PRED, NEW_PRED)

# Add report key
OLD_RPT = "                    'unknown0_lgbm_decision_metrics': grouped_prediction_metrics(pred_df, 'unknown0_mlp_pred'),\n"
NEW_RPT = (
    "                    'unknown0_lgbm_decision_metrics': grouped_prediction_metrics(pred_df, 'unknown0_mlp_pred'),\n"
    "                    'unknown0_lgbm_decision_metrics_infer_faithful': UNKNOWN0_INFER_FAITHFUL_METRICS,\n"
)
assert OLD_RPT in src19, 'report metrics anchor not found in Cell[19]'
src19 = src19.replace(OLD_RPT, NEW_RPT)

cells[19]['source'] = src19.splitlines(keepends=True)

# ── Clear outputs ─────────────────────────────────────────────────────────────
for c in cells:
    if c.get('cell_type') == 'code':
        c['outputs'] = []
        c['execution_count'] = None
    else:
        c.pop('outputs', None)
        c.pop('execution_count', None)

with open(DST, 'w') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f'Written: {DST}')

# ── Sanity checks ─────────────────────────────────────────────────────────────
with open(DST) as f:
    nb2 = json.load(f)
s1  = ''.join(nb2['cells'][1]['source'])
s7  = ''.join(nb2['cells'][7]['source'])
s19 = ''.join(nb2['cells'][19]['source'])

assert 'v0_09d1_infer_faithful' in s1
assert "RUN_PREFIX = 'pokemon-20260627-v0-09d1'" in s1
assert 'v08d19' in s7
assert 'INFER-FAITHFUL' in s19
assert 'unknown0_lgbm_decision_metrics_infer_faithful' in s19
assert 'op_active_energy_count' in s19
assert "lambdarank_truncation_level': 1" in s19, 'trunc1 must be kept'
assert 'position_winrate' in s19
assert '_winner_weight = 4.0' in s19
full = ''.join(''.join(c['source']) for c in nb2['cells'])
assert 'lambdarank' in full
print('All sanity checks passed.')
