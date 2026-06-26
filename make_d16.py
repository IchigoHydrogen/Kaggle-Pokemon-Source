"""Create v05d16 notebook from v05d15:
  - Top200 episode filter (T200_T200 only)
  - Phase-aware weight (early=0.5, mid=1.0, late=0.2, replaces MC discount)
  - Delta-V N-step lookahead (N=3) for advantage filtering
  - UNKNOWN_0 MLP direct in submission (torch-based, falls back to policy table)
"""
import json
from pathlib import Path

WORK_DIR = Path('/kaggle/working')
SRC_NB = WORK_DIR / 'pokemon-20260625-v0-05d15.ipynb'
DST_NB = WORK_DIR / 'pokemon-20260625-v0-05d16.ipynb'

with open(SRC_NB) as f:
    nb = json.load(f)

def cell_src(i):
    return ''.join(nb['cells'][i]['source'])

def set_cell_src(i, s):
    nb['cells'][i]['source'] = s
    nb['cells'][i]['outputs'] = []
    if nb['cells'][i].get('cell_type') == 'code':
        nb['cells'][i]['execution_count'] = None

def insert_cell_after(i, source):
    """Insert a new code cell after cell i."""
    new_cell = {
        'cell_type': 'code',
        'execution_count': None,
        'metadata': {},
        'outputs': [],
        'source': source,
    }
    nb['cells'].insert(i + 1, new_cell)

# ── Cell 1: experiment config ───────────────────────────────────────────────
src1 = cell_src(1)
src1 = src1.replace(
    "EXPERIMENT_NAME = 'v0_05d15_to_bench_scoring'",
    "EXPERIMENT_NAME = 'v0_05d16_adv_top200_mlp'"
)
src1 = src1.replace(
    "RUN_PREFIX = 'pokemon-20260625-v0-05d15'",
    "RUN_PREFIX = 'pokemon-20260625-v0-05d16'"
)
# Change FINAL_SUBMISSION_VARIANT to use MLP direct
src1 = src1.replace(
    "FINAL_SUBMISSION_VARIANT = os.environ.get('V05_FINAL_SUBMISSION_VARIANT', 'unknown0_policy_table').strip()",
    "FINAL_SUBMISSION_VARIANT = os.environ.get('V05_FINAL_SUBMISSION_VARIANT', 'unknown0_mlp_direct').strip()"
)
# Add v05d16-specific flags after MLP_ALPHA line
OLD_FLAGS = "MLP_ALPHA = float(os.environ.get('V05_MLP_ALPHA', '0.10'))"
NEW_FLAGS = """MLP_ALPHA = float(os.environ.get('V05_MLP_ALPHA', '1.00'))
# v05d16: training data quality improvements
TOP200_EPISODE_FILTER = os.environ.get('V05_TOP200_FILTER', '1') != '0'
USE_PHASE_AWARE_WEIGHT = os.environ.get('V05_PHASE_WEIGHT', '1') != '0'
N_STEP_LOOKAHEAD = int(os.environ.get('V05_N_STEP', '3'))"""
if OLD_FLAGS in src1:
    src1 = src1.replace(OLD_FLAGS, NEW_FLAGS)
    print("Cell 1: flags patched")
else:
    print("WARNING: Cell 1 flag target not found")
set_cell_src(1, src1)

assert 'v0_05d16_adv_top200_mlp' in cell_src(1), "Cell 1 EXPERIMENT_NAME patch failed"
assert 'pokemon-20260625-v0-05d16' in cell_src(1), "Cell 1 RUN_PREFIX patch failed"
assert 'TOP200_EPISODE_FILTER' in cell_src(1), "Cell 1 TOP200_EPISODE_FILTER patch failed"
print("Cell 1: OK")

# ── Cell 7: phase-aware weight + Top200 filter ──────────────────────────────
src7 = cell_src(7)

OLD_WEIGHT = (
    "    DECISION_ROWS_DF['mc_step_weight'] = MC_DISCOUNT_GAMMA ** (_T - _step).clip(lower=0)\n"
    "    DECISION_ROWS_DF = DECISION_ROWS_DF.drop(columns=['_mc_n_steps'])\n"
    "    _w = DECISION_ROWS_DF['mc_step_weight']\n"
    "    print(f'mc_step_weight added: min={_w.min():.4f} mean={_w.mean():.4f} max={_w.max():.4f}')\n"
    "else:\n"
    "    DECISION_ROWS_DF['mc_step_weight'] = 1.0\n"
    "    print('mc_step_weight: uniform 1.0')"
)

NEW_WEIGHT = (
    "    if USE_PHASE_AWARE_WEIGHT:\n"
    "        _frac = (_step / _T.clip(lower=1)).clip(0, 1)\n"
    "        DECISION_ROWS_DF['step_frac'] = _frac\n"
    "        _pw = _frac.map(lambda x: 0.5 if x < 1/3 else (1.0 if x <= 2/3 else 0.2)).astype(float)\n"
    "        DECISION_ROWS_DF['phase_weight'] = _pw\n"
    "        DECISION_ROWS_DF['mc_step_weight'] = _pw\n"
    "    else:\n"
    "        DECISION_ROWS_DF['mc_step_weight'] = MC_DISCOUNT_GAMMA ** (_T - _step).clip(lower=0)\n"
    "        DECISION_ROWS_DF['phase_weight'] = DECISION_ROWS_DF['mc_step_weight']\n"
    "    DECISION_ROWS_DF = DECISION_ROWS_DF.drop(columns=['_mc_n_steps'])\n"
    "    _w = DECISION_ROWS_DF['mc_step_weight']\n"
    "    print(f'mc_step_weight (phase_aware={USE_PHASE_AWARE_WEIGHT}): "
        "min={_w.min():.4f} mean={_w.mean():.4f} max={_w.max():.4f}')\n"
    "else:\n"
    "    DECISION_ROWS_DF['mc_step_weight'] = 1.0\n"
    "    DECISION_ROWS_DF['phase_weight'] = 1.0\n"
    "    print('mc_step_weight: uniform 1.0')\n"
    "\n"
    "# v05d16: Top200 episode filter\n"
    "if TOP200_EPISODE_FILTER and not DECISION_ROWS_DF.empty and 'tier' in DECISION_ROWS_DF.columns:\n"
    "    _n_before = len(DECISION_ROWS_DF)\n"
    "    DECISION_ROWS_DF = DECISION_ROWS_DF[DECISION_ROWS_DF['tier'] == 'T200_T200'].copy()\n"
    "    print(f'T200_T200 filter: {_n_before} → {len(DECISION_ROWS_DF)} rows')"
)

if OLD_WEIGHT in src7:
    src7 = src7.replace(OLD_WEIGHT, NEW_WEIGHT)
    print("Cell 7: weight+filter patched")
else:
    print("WARNING: Cell 7 weight target not found!")
    idx = src7.find("DECISION_ROWS_DF['mc_step_weight'] = MC_DISCOUNT_GAMMA")
    print(f"  mc_step_weight assignment found at: {idx}")

set_cell_src(7, src7)
assert 'TOP200_EPISODE_FILTER' in cell_src(7), "Cell 7 Top200 filter patch failed"
assert 'USE_PHASE_AWARE_WEIGHT' in cell_src(7), "Cell 7 phase_weight patch failed"
print("Cell 7: OK")

# ── Cell 9: propagate mc_step_weight through to ALAKAZAM_OPTION_MODEL_DF ───
src9 = cell_src(9)

OLD_DCOLS = (
    "    dcols = [\n"
    "        'decision_id', 'episode_id', 'player_index', 'rank', 'rate', 'is_top200',\n"
    "        'player_archetype', 'opponent_archetype', 'tier', 'matchup', 'context_name',\n"
    "        'min_count', 'max_count', 'num_options', 'chosen_valid', 'won', 'chosen_count'\n"
    "    ]"
)

NEW_DCOLS = (
    "    dcols = [\n"
    "        'decision_id', 'episode_id', 'player_index', 'rank', 'rate', 'is_top200',\n"
    "        'player_archetype', 'opponent_archetype', 'tier', 'matchup', 'context_name',\n"
    "        'min_count', 'max_count', 'num_options', 'chosen_valid', 'won', 'chosen_count',\n"
    "        'step', 'mc_step_weight', 'phase_weight', 'step_frac',\n"
    "    ]"
)

if OLD_DCOLS in src9:
    src9 = src9.replace(OLD_DCOLS, NEW_DCOLS)
    print("Cell 9: dcols patched")
else:
    print("WARNING: Cell 9 dcols target not found!")
    idx = src9.find("'decision_id', 'episode_id', 'player_index',")
    print(f"  dcols found at: {idx}")

set_cell_src(9, src9)
assert 'mc_step_weight' in cell_src(9), "Cell 9 mc_step_weight patch failed"
print("Cell 9: OK")

# ── Find Cell 12 index ──────────────────────────────────────────────────────
# Cell 12 is the value model training cell; find it by content
cell12_idx = None
for i, cell in enumerate(nb['cells']):
    src = ''.join(cell['source'])
    if 'train_value_model' in src and 'ALAKAZAM_VALUE_MODEL' in src and 'predict_proba' in src:
        cell12_idx = i
        break
if cell12_idx is None:
    raise RuntimeError("Could not find Cell 12 (value model training)")
print(f"Cell 12 found at index {cell12_idx}")

# ── New cell after Cell 12: Delta-V N-step lookahead ───────────────────────
DELTA_V_CELL = """\
# ── v05d16: Delta-V N-step lookahead + mc_step_weight update ────────────────
N_STEP = N_STEP_LOOKAHEAD  # default=3
DELTA_V_REPORT = {'status': 'not_run'}

if ALAKAZAM_VALUE_MODEL is not None and not ALAKAZAM_OPTION_MODEL_DF.empty:
    try:
        _val_mdl = ALAKAZAM_VALUE_MODEL['model']
        _vnum = ALAKAZAM_VALUE_MODEL.get('numeric_features', VALUE_NUMERIC_FEATURES)
        _vcat = ALAKAZAM_VALUE_MODEL.get('categorical_features', VALUE_CATEGORICAL_FEATURES)
        _dec = ALAKAZAM_OPTION_MODEL_DF.drop_duplicates('decision_id').copy()
        _avail = [c for c in _vnum + _vcat if c in _dec.columns]
        _dec['v_t'] = _val_mdl.predict_proba(_dec[_avail])[:, 1]
        if 'step' in _dec.columns:
            _dec = _dec.sort_values(['episode_id', 'player_index', 'step'])
            _dec['v_t_plus_N'] = _dec.groupby(['episode_id', 'player_index'])['v_t'].shift(-N_STEP)
            _dec['delta_v'] = _dec['v_t_plus_N'] - _dec['v_t']
        else:
            _dec['delta_v'] = float('nan')
        _dec['delta_v_weight'] = _dec['delta_v'].clip(lower=0).fillna(0.0)
        _pw_col = 'phase_weight' if 'phase_weight' in _dec.columns else 'mc_step_weight'
        _dec['combined_weight'] = _dec.get(_pw_col, pd.Series(0.5, index=_dec.index)).fillna(0.5) * _dec['delta_v_weight']
        _wmap = _dec.set_index('decision_id')['combined_weight'].to_dict()
        DECISION_ROWS_DF['mc_step_weight'] = DECISION_ROWS_DF['decision_id'].map(_wmap).fillna(0.0)
        ALAKAZAM_OPTION_MODEL_DF['mc_step_weight'] = ALAKAZAM_OPTION_MODEL_DF['decision_id'].map(_wmap).fillna(0.0)
        _n_pos = (_dec['combined_weight'] > 0).sum()
        _dv_pos = (_dec['delta_v'].fillna(-1) > 0).mean()
        DELTA_V_REPORT = {
            'status': 'ok', 'N_step': N_STEP,
            'n_decisions': int(len(_dec)),
            'positive_weight': int(_n_pos),
            'delta_v_mean': float(_dec['delta_v'].mean()),
            'delta_v_positive_pct': float(_dv_pos),
        }
        print(f'Delta-V (N={N_STEP}): decisions={len(_dec)}, positive_dv={100*_dv_pos:.1f}%, weight>0={_n_pos}/{len(_dec)}')
        print(f'  delta_v: mean={_dec["delta_v"].mean():.4f} std={_dec["delta_v"].std():.4f}')
        with open(OUTPUT_DIR / 'delta_v_report.json', 'w') as _f:
            json.dump(DELTA_V_REPORT, _f, indent=2)
        ARTIFACT_PATHS['delta_v_report'] = str(OUTPUT_DIR / 'delta_v_report.json')
    except Exception as _exc:
        import traceback as _tb
        DELTA_V_REPORT = {'status': 'error', 'error': repr(_exc)}
        print(f'Delta-V computation failed: {_exc}')
        print(_tb.format_exc(limit=4))
else:
    print('Delta-V skipped (no value model or empty ALAKAZAM_OPTION_MODEL_DF)')
    DELTA_V_REPORT = {'status': 'skipped', 'reason': 'no_value_model_or_empty_data'}
print('DELTA_V_REPORT:', DELTA_V_REPORT.get('status'))
"""

insert_cell_after(cell12_idx, DELTA_V_CELL)
print(f"Delta-V cell inserted after index {cell12_idx}")

# ── Cell 18: add make_unknown0_mlp_direct_submission_agent_source ───────────
# Find Cell 18 by content (after possible index shift from inserted cell)
cell18_idx = None
for i, cell in enumerate(nb['cells']):
    src = ''.join(cell['source'])
    if 'UNKNOWN0_MLP_REPORT' in src and 'Unknown0OptionMLP' in src:
        cell18_idx = i
        break
if cell18_idx is None:
    raise RuntimeError("Could not find Cell 18 (Unknown0 MLP)")
print(f"Cell 18 found at index {cell18_idx}")

src18 = cell_src(cell18_idx)

# Injection code that goes into main.py for MLP inference
MLP_INJECTION_CODE = r'''
# v0-05d16: UNKNOWN_0 direct MLP inference (torch). Falls back to policy table on failure.
import math as _u0_math

_U0_MLP_BUNDLE = None

def _u0_load_mlp():
    global _U0_MLP_BUNDLE
    try:
        import os, torch
        import torch.nn as _u0_nn
        _d = os.path.dirname(os.path.abspath(__file__))
        _p = os.path.join(_d, 'unknown0_mlp.pt')
        if not os.path.exists(_p):
            return
        _b = torch.load(_p, map_location='cpu', weights_only=False)
        class _U0MLP(_u0_nn.Module):
            def __init__(self, nd, cc):
                super().__init__()
                self.embs = _u0_nn.ModuleList()
                et = 0
                for c in cc:
                    d = int(min(24, max(4, round(_u0_math.sqrt(max(c, 2))))))
                    self.embs.append(_u0_nn.Embedding(c, d))
                    et += d
                self.net = _u0_nn.Sequential(
                    _u0_nn.Linear(int(nd + et), 96), _u0_nn.ReLU(), _u0_nn.Dropout(0.1),
                    _u0_nn.Linear(96, 48), _u0_nn.ReLU(), _u0_nn.Dropout(0.05),
                    _u0_nn.Linear(48, 1),
                )
            def forward(self, xn, xc):
                parts = [xn] + [e(xc[:, j]) for j, e in enumerate(self.embs)]
                return self.net(torch.cat(parts, 1)).squeeze(1)
        _m = _U0MLP(_b['num_dim'], _b['cat_cardinalities'])
        _m.load_state_dict(_b['state_dict'])
        _m.eval()
        _U0_MLP_BUNDLE = {'model': _m, 'prep': _b['prep']}
    except Exception:
        pass

_u0_load_mlp()


def _u0_infer_scores(obs, select, my_index, pi_opp):
    """Return per-option MLP logits for UNKNOWN_0, or None on failure."""
    if _U0_MLP_BUNDLE is None:
        return None
    try:
        import numpy as _np, torch as _tc
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
        prep = _U0_MLP_BUNDLE['prep']
        num_arr = []
        for c in prep.get('numeric_cols', []):
            vals = [float(r.get(c, 0) or 0) for r in rows]
            st = prep['num_stats'].get(c, {'mean': 0.0, 'std': 1.0})
            mn = float(st.get('mean', 0) or 0); sd = float(st.get('std', 1) or 1)
            if sd < 1e-6: sd = 1.0
            num_arr.append(_np.array([(v - mn) / sd for v in vals], dtype='float32'))
        X_num = (_np.vstack(num_arr).T.astype('float32') if num_arr
                 else _np.zeros((len(rows), 0), dtype='float32'))
        cat_arr = []
        for c in prep.get('categorical_cols', []):
            mapping = prep.get('cat_maps', {}).get(c, {})
            vals = [str(r.get(c, 'UNK') or 'UNK') for r in rows]
            cat_arr.append(_np.array([mapping.get(v, 0) for v in vals], dtype='int64'))
        X_cat = (_np.vstack(cat_arr).T.astype('int64') if cat_arr
                 else _np.zeros((len(rows), 0), dtype='int64'))
        with _tc.no_grad():
            logits = _U0_MLP_BUNDLE['model'](
                _tc.tensor(X_num, dtype=_tc.float32),
                _tc.tensor(X_cat, dtype=_tc.long),
            ).numpy()
        return logits.tolist()
    except Exception:
        return None
'''

# Patch site: inject MLP early-return BEFORE unknown0_policy_selected assignment
MLP_CALL_SITE_PATCH_TARGET = "    unknown0_policy_selected = _unknown0_policy_select("
MLP_CALL_SITE_PATCH_REPLACEMENT = (
    "    if _U0_MLP_BUNDLE is not None and _unknown0_policy_is_context(context):\n"
    "        _mlp_logits = _u0_infer_scores(obs, select, my_index, 1 - my_index)\n"
    "        if _mlp_logits and len(_mlp_logits) == len(select.option):\n"
    "            _best_i = max(range(len(_mlp_logits)), key=lambda _i: _mlp_logits[_i])\n"
    "            _sel = safe_unique_action([_best_i], len(select.option), min_count, max_count)\n"
    "            if len(_sel) >= min_count:\n"
    "                return _sel\n"
    "    unknown0_policy_selected = _unknown0_policy_select("
)

def make_unknown0_mlp_direct_submission_agent_source(policy_table_source):
    """Inject UNKNOWN_0 MLP inference into policy_table_source."""
    source = policy_table_source
    # 1. Add import torch at the top (after last import line before first def/class)
    import_line = 'import torch\nimport numpy as _u0np\n'
    # Find insertion point: after the last top-level import
    # Simple approach: inject after the policy table helper code block
    # Find a safe insertion point: before def agent(
    agent_def = 'def agent(obs_dict: dict) -> list[int]:'
    if agent_def not in source:
        return source  # can't inject
    idx = source.find(agent_def)
    source = source[:idx] + MLP_INJECTION_CODE + '\n\n' + source[idx:]
    # 2. Patch the unknown0_policy_select call site
    if MLP_CALL_SITE_PATCH_TARGET in source:
        source = source.replace(
            MLP_CALL_SITE_PATCH_TARGET,
            MLP_CALL_SITE_PATCH_REPLACEMENT
        )
    return source


# Append the function definition to Cell 18
CELL18_ADDITION = f'''

# ── v05d16: MLP direct submission source builder ───────────────────────────
{make_unknown0_mlp_direct_submission_agent_source.__doc__ and ""}
import inspect as _inspect_mod
exec(compile(_inspect_mod.getsource(make_unknown0_mlp_direct_submission_agent_source), '<cell18>', 'exec'))

'''

# Actually, we need to embed the function SOURCE in the notebook cell.
# Build the Cell 18 addition as a string literal
MLP_INJECTION_CODE_ESCAPED = MLP_INJECTION_CODE.replace("'", "\\'").replace('"', '\\"')

CELL18_ADDITION_STR = '''

# ── v05d16: UNKNOWN_0 MLP direct submission source builder ──────────────────
_MLP_INJECTION_CODE = """ + repr(MLP_INJECTION_CODE) + """

_MLP_CALL_PATCH_TARGET = '    unknown0_policy_selected = _unknown0_policy_select('
_MLP_CALL_PATCH_REPLACEMENT = (
    '    if _U0_MLP_BUNDLE is not None and _unknown0_policy_is_context(context):\\n'
    '        _mlp_logits = _u0_infer_scores(obs, select, my_index, 1 - my_index)\\n'
    '        if _mlp_logits and len(_mlp_logits) == len(select.option):\\n'
    '            _best_i = max(range(len(_mlp_logits)), key=lambda _i: _mlp_logits[_i])\\n'
    '            _sel = safe_unique_action([_best_i], len(select.option), min_count, max_count)\\n'
    '            if len(_sel) >= min_count:\\n'
    '                return _sel\\n'
    '    unknown0_policy_selected = _unknown0_policy_select('
)

def make_unknown0_mlp_direct_submission_agent_source(policy_table_source):
    source = policy_table_source
    agent_def = 'def agent(obs_dict: dict) -> list[int]:'
    if agent_def not in source:
        return source
    idx = source.find(agent_def)
    source = source[:idx] + _MLP_INJECTION_CODE + '\\n\\n' + source[idx:]
    if _MLP_CALL_PATCH_TARGET in source:
        source = source.replace(_MLP_CALL_PATCH_TARGET, _MLP_CALL_PATCH_REPLACEMENT)
    return source

UNKNOWN0_DIRECT_MLP_FUNC_STATUS = {'status': 'ok'}
print('make_unknown0_mlp_direct_submission_agent_source: defined')
'''

# The above approach is too complex for string escaping. Use a clean approach:
# Embed the function as a proper Python source string in the notebook.

CELL18_ADDITION_CLEAN = (
    "\n"
    "\n# ── v05d16: UNKNOWN_0 MLP direct submission source builder ──────────────────\n"
    "_MLP_INJ_CODE = " + repr(MLP_INJECTION_CODE) + "\n"
    "\n"
    "_MLP_PATCH_TARGET = '    unknown0_policy_selected = _unknown0_policy_select('\n"
    "_MLP_PATCH_REPL = (\n"
    "    '    if _U0_MLP_BUNDLE is not None and _unknown0_policy_is_context(context):\\n'\n"
    "    '        _mlp_logits = _u0_infer_scores(obs, select, my_index, 1 - my_index)\\n'\n"
    "    '        if _mlp_logits and len(_mlp_logits) == len(select.option):\\n'\n"
    "    '            _best_i = max(range(len(_mlp_logits)), key=lambda _i: _mlp_logits[_i])\\n'\n"
    "    '            _sel = safe_unique_action([_best_i], len(select.option), min_count, max_count)\\n'\n"
    "    '            if len(_sel) >= min_count:\\n'\n"
    "    '                return _sel\\n'\n"
    "    '    unknown0_policy_selected = _unknown0_policy_select('\n"
    ")\n"
    "\n"
    "def make_unknown0_mlp_direct_submission_agent_source(policy_table_source):\n"
    "    source = policy_table_source\n"
    "    marker = 'def agent(obs_dict: dict) -> list[int]:'\n"
    "    if marker not in source:\n"
    "        return source\n"
    "    idx = source.find(marker)\n"
    "    source = source[:idx] + _MLP_INJ_CODE + '\\n\\n' + source[idx:]\n"
    "    if _MLP_PATCH_TARGET in source:\n"
    "        source = source.replace(_MLP_PATCH_TARGET, _MLP_PATCH_REPL)\n"
    "    return source\n"
    "\n"
    "print('make_unknown0_mlp_direct_submission_agent_source: defined')\n"
)

src18 = src18 + CELL18_ADDITION_CLEAN
set_cell_src(cell18_idx, src18)
assert 'make_unknown0_mlp_direct_submission_agent_source' in cell_src(cell18_idx), "Cell 18 MLP builder patch failed"
print("Cell 18: OK")

# ── Cell 20: remove torch assertions + add MLP direct variant ───────────────
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

# 1. Remove torch assertions (3 different patterns × 1 each)
TORCH_ASSERT_1 = (
    "    if 'import torch' in main_source:\n"
    "        raise AssertionError('submission main.py must not import torch')"
)
TORCH_ASSERT_2 = (
    "    if 'import torch' in main_source:\n"
    "        raise AssertionError('final submission main.py must not import torch')"
)
TORCH_ASSERT_3 = (
    "if 'import torch' in FINAL_MAIN_SOURCE:\n"
    "    raise AssertionError('final selected submission cannot import torch')"
)

for old, new in [
    (TORCH_ASSERT_1, "    # torch allowed in v05d16: UNKNOWN_0 MLP uses torch"),
    (TORCH_ASSERT_2, "    # torch allowed in v05d16: UNKNOWN_0 MLP uses torch"),
    (TORCH_ASSERT_3, "# torch check removed in v05d16: UNKNOWN_0 MLP uses torch"),
]:
    if old in src20:
        src20 = src20.replace(old, new)
        print(f"Cell 20: removed torch assertion: {repr(old[:60])}")
    else:
        print(f"WARNING: torch assertion not found: {repr(old[:60])}")

# 2. Modify build_submission_archive to accept extra_files
OLD_FUNC_SIG = "def build_submission_archive(main_source: str, deck: list[int], archive_path: str | Path = 'submission.tar.gz') -> dict:"
NEW_FUNC_SIG = "def build_submission_archive(main_source: str, deck: list[int], archive_path: str | Path = 'submission.tar.gz', extra_files=None) -> dict:"

if OLD_FUNC_SIG in src20:
    src20 = src20.replace(OLD_FUNC_SIG, NEW_FUNC_SIG)
    print("Cell 20: build_submission_archive signature patched")
else:
    print("WARNING: build_submission_archive signature not found!")

# 3. Add extra_files to tarfile building
OLD_TAR_BLOCK = (
    "        tar.add(str(cg_dir), arcname='cg', filter=exclude_python_cache)\n"
    "    with tarfile.open(archive_path, 'r:gz') as tar:"
)
NEW_TAR_BLOCK = (
    "        tar.add(str(cg_dir), arcname='cg', filter=exclude_python_cache)\n"
    "        for _ef_src, _ef_arc in (extra_files or []):\n"
    "            if Path(str(_ef_src)).exists():\n"
    "                tar.add(str(_ef_src), arcname=_ef_arc)\n"
    "    with tarfile.open(archive_path, 'r:gz') as tar:"
)
if OLD_TAR_BLOCK in src20:
    src20 = src20.replace(OLD_TAR_BLOCK, NEW_TAR_BLOCK)
    print("Cell 20: tarfile extra_files patched")
else:
    print("WARNING: tarfile block not found!")

# 4. Add MLP direct variant building + updated SUBMISSION_VARIANT_SOURCES dict
OLD_VARIANT_SOURCES = (
    "SUBMISSION_VARIANT_SOURCES = {\n"
    "    'rule_only': RULE_ONLY_MAIN_SOURCE,\n"
    "    'unknown0_policy_table': UNKNOWN0_POLICY_MAIN_SOURCE,\n"
    "}\n"
    "SUBMISSION_VARIANT_ARCHIVES = {\n"
    "    'rule_only': f'{RUN_PREFIX}-submission-rule-only.tar.gz',\n"
    "    'unknown0_policy_table': f'{RUN_PREFIX}-submission-unknown0-policy-table.tar.gz',\n"
    "}"
)

NEW_VARIANT_SOURCES = (
    "# v05d16: Build MLP direct submission source\n"
    "UNKNOWN0_MLP_DIRECT_MAIN_SOURCE = UNKNOWN0_POLICY_MAIN_SOURCE\n"
    "UNKNOWN0_MLP_DIRECT_STATUS = {'status': 'not_built'}\n"
    "_u0_model_pt = MODEL_DIR / f'{RUN_PREFIX}-unknown0_mlp_reranker.pt'\n"
    "if 'make_unknown0_mlp_direct_submission_agent_source' in globals() and _u0_model_pt.exists():\n"
    "    try:\n"
    "        UNKNOWN0_MLP_DIRECT_MAIN_SOURCE = make_unknown0_mlp_direct_submission_agent_source(\n"
    "            UNKNOWN0_POLICY_MAIN_SOURCE\n"
    "        )\n"
    "        UNKNOWN0_MLP_DIRECT_STATUS = {'status': 'ok', 'model': str(_u0_model_pt)}\n"
    "        print(f'MLP direct source built, len={len(UNKNOWN0_MLP_DIRECT_MAIN_SOURCE)}')\n"
    "        assert '_U0_MLP_BUNDLE' in UNKNOWN0_MLP_DIRECT_MAIN_SOURCE, 'MLP code not injected'\n"
    "        assert '_u0_infer_scores' in UNKNOWN0_MLP_DIRECT_MAIN_SOURCE, 'infer function missing'\n"
    "    except Exception as _exc:\n"
    "        UNKNOWN0_MLP_DIRECT_STATUS = {'status': 'error', 'error': repr(_exc)}\n"
    "        print(f'MLP direct source build failed: {_exc}')\n"
    "        UNKNOWN0_MLP_DIRECT_MAIN_SOURCE = UNKNOWN0_POLICY_MAIN_SOURCE\n"
    "else:\n"
    "    UNKNOWN0_MLP_DIRECT_STATUS = {'status': 'skipped',\n"
    "        'reason': f'model_missing={not _u0_model_pt.exists()}'}\n"
    "    print(f'MLP direct: skipped ({UNKNOWN0_MLP_DIRECT_STATUS[\"reason\"]})')\n"
    "\n"
    "SUBMISSION_VARIANT_SOURCES = {\n"
    "    'rule_only': RULE_ONLY_MAIN_SOURCE,\n"
    "    'unknown0_policy_table': UNKNOWN0_POLICY_MAIN_SOURCE,\n"
    "    'unknown0_mlp_direct': UNKNOWN0_MLP_DIRECT_MAIN_SOURCE,\n"
    "}\n"
    "SUBMISSION_VARIANT_ARCHIVES = {\n"
    "    'rule_only': f'{RUN_PREFIX}-submission-rule-only.tar.gz',\n"
    "    'unknown0_policy_table': f'{RUN_PREFIX}-submission-unknown0-policy-table.tar.gz',\n"
    "    'unknown0_mlp_direct': f'{RUN_PREFIX}-submission-unknown0-mlp-direct.tar.gz',\n"
    "}\n"
    "SUBMISSION_VARIANT_EXTRA_FILES = {\n"
    "    'rule_only': [],\n"
    "    'unknown0_policy_table': [],\n"
    "    'unknown0_mlp_direct': ([(str(_u0_model_pt), 'unknown0_mlp.pt')]\n"
    "                            if _u0_model_pt.exists() else []),\n"
    "}"
)

if OLD_VARIANT_SOURCES in src20:
    src20 = src20.replace(OLD_VARIANT_SOURCES, NEW_VARIANT_SOURCES)
    print("Cell 20: SUBMISSION_VARIANT_SOURCES patched")
else:
    print("WARNING: SUBMISSION_VARIANT_SOURCES not found!")
    # Debug: find the dict start
    idx = src20.find("SUBMISSION_VARIANT_SOURCES = {")
    print(f"  Found at {idx}: {repr(src20[idx:idx+200])}")

# 5. Pass extra_files in the variant loop
OLD_LOOP_CALL = (
    "            summary = build_submission_archive(source, SELECTED_ALAKAZAM_DECK, SUBMISSION_VARIANT_ARCHIVES[variant])"
)
NEW_LOOP_CALL = (
    "            _ef = (SUBMISSION_VARIANT_EXTRA_FILES.get(variant, [])\n"
    "                   if 'SUBMISSION_VARIANT_EXTRA_FILES' in globals() else [])\n"
    "            summary = build_submission_archive(source, SELECTED_ALAKAZAM_DECK,\n"
    "                                               SUBMISSION_VARIANT_ARCHIVES[variant],\n"
    "                                               extra_files=_ef)"
)
if OLD_LOOP_CALL in src20:
    src20 = src20.replace(OLD_LOOP_CALL, NEW_LOOP_CALL)
    print("Cell 20: variant loop call patched")
else:
    print("WARNING: variant loop call not found!")

set_cell_src(cell20_idx, src20)

# Verify all torch assertions removed
remaining = src20.count("raise AssertionError('submission main.py must not import torch')")
remaining += src20.count("raise AssertionError('final submission main.py must not import torch')")
remaining += src20.count("raise AssertionError('final selected submission cannot import torch')")
if remaining > 0:
    print(f"WARNING: {remaining} torch assertions still remain!")
else:
    print("Cell 20: all torch assertions removed")
assert 'UNKNOWN0_MLP_DIRECT_MAIN_SOURCE' in cell_src(cell20_idx), "Cell 20 MLP direct variant missing"
print("Cell 20: OK")

# ── Clear all outputs ────────────────────────────────────────────────────────
for cell in nb['cells']:
    if cell.get('cell_type') == 'code':
        cell['outputs'] = []
        cell['execution_count'] = None

# ── Write notebook ───────────────────────────────────────────────────────────
with open(DST_NB, 'w') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print(f"\nv05d16 notebook written: {DST_NB}")
print(f"Total cells: {len(nb['cells'])}")
