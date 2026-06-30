"""Build pokemon-20260627-v0-08d19-remote-pc.ipynb from v08d18.

v08d19: sequential opponent features — what did opponent do last turn?

Adds opponent last-action context as a feature by computing it from DECISION_ROWS_DF
using step ordering within each episode. No re-mining required (preloads v08d18).

Same data split as v08d18 (preload) → proper apples-to-apples comparison for
evaluating whether sequential opponent features improve top1 over just prize_gap.

New features (computed in Cell[19] from DECISION_ROWS_DF):
  op_last_context   (cat): context_name of opponent's most recent decision before ours
                           (UNKNOWN_0, MAIN, ATTACH_FROM, SWITCH, etc. or NONE)
  steps_since_op   (num): step_T - op_last_step  (how recently opponent acted)

Inference:
  op_last_context is approximated from observable state changes:
    - if op_active_energy increased   → opponent attached energy (ATTACH_FROM-like)
    - if my_active_hp decreased vs prior (we can't track this easily, so use UNKNOWN)
  Simplification: at inference we track op_last_context via a closure variable
  that records the context after each observed opponent-turn state.
  But for the first implementation, use a simpler proxy:
    op_last_context_inferred = 'UNKNOWN_0' if op_active_energy_count >= 2 else 'ATTACH_FROM'
  This is an approximation; exact inference requires agent-side state tracking.

Changes from v08d18:
  1. EXPERIMENT_NAME → v0_08d19_op_seq
  2. RUN_PREFIX → pokemon-20260627-v0-08d19
  3. Cell[7]: Preload from v08d18 (same split as v08d18 for fair comparison)
  4. Cell[19]: Add op_last_context + steps_since_op via merge_asof on DECISION_ROWS_DF
  5. Cell[19]: Add op_last_context to categorical features
  6. Cell[19]: Add steps_since_op to numeric (via extra_numeric_candidates)
  7. Cell[19]: Injection code uses simple proxy for op_last_context at inference
"""
import json

SRC = '/kaggle/working/pokemon-20260627-v0-08d18.ipynb'
DST = '/kaggle/working/pokemon-20260627-v0-08d19.ipynb'

with open(SRC) as f:
    nb = json.load(f)

cells = nb['cells']

# ── Cell[1]: Config ───────────────────────────────────────────────────────────
src1 = ''.join(cells[1]['source'])

OLD_EXP = "EXPERIMENT_NAME = 'v0_08d18_prize_gap'"
NEW_EXP = "EXPERIMENT_NAME = 'v0_08d19_op_seq'"
assert OLD_EXP in src1, f'EXPERIMENT_NAME not found'
src1 = src1.replace(OLD_EXP, NEW_EXP)

OLD_PFX = "RUN_PREFIX = 'pokemon-20260627-v0-08d18'"
NEW_PFX = "RUN_PREFIX = 'pokemon-20260627-v0-08d19'"
assert OLD_PFX in src1
src1 = src1.replace(OLD_PFX, NEW_PFX)

cells[1]['source'] = src1.splitlines(keepends=True)

# ── Cell[7]: Preload from v08d18 (has correct prizes_left) ───────────────────
src7 = ''.join(cells[7]['source'])

OLD_NO_PRELOAD = (
    "# v08d18: NO preload — prizes_left bug fix requires fresh mining\n"
    "# Preloaded parquets have prizes_left=0 (old bug). Must re-mine.\n"
    "print('v08d18: forcing fresh episode mining (prizes_left fix requires it)')\n\n"
    "if RUN_REPLAY_MINING:"
)
NEW_PRELOAD = (
    "# v08d19: preload from v08d18 (has correct prizes_left after bug fix)\n"
    "_PRELOAD_CANDIDATES = [\n"
    "    Path('/kaggle/working/pokemon-20260627-v0-08d18/pokemon-20260627-v0-08d18'),\n"
    "]\n"
    "_PRELOAD_PFX = None\n"
    "for _cand in _PRELOAD_CANDIDATES:\n"
    "    if (_cand.parent.exists() and\n"
    "            (_cand.parent / f'{_cand.name}-decision_rows.parquet').exists()):\n"
    "        _PRELOAD_PFX = str(_cand)\n"
    "        break\n"
    "_preload_ok = _PRELOAD_PFX is not None\n"
    "if _preload_ok:\n"
    "    _pfx = _PRELOAD_PFX\n"
    "    DECISION_ROWS_DF  = pd.read_parquet(f'{_pfx}-decision_rows.parquet')\n"
    "    OPTION_ROWS_DF    = pd.read_parquet(f'{_pfx}-option_rows.parquet')\n"
    "    EPISODE_INDEX_DF  = pd.read_parquet(f'{_pfx}-episode_index.parquet')\n"
    "    EPISODE_SPLIT_DF  = pd.read_parquet(f'{_pfx}-episode_split.parquet')\n"
    "    DECKLISTS_DF      = pd.read_parquet(f'{_pfx}-decklists.parquet')\n"
    "    STATE_SUMMARY_DF  = pd.read_parquet(f'{_pfx}-state_summary.parquet')\n"
    "    RUN_REPLAY_MINING = False\n"
    "    print(f'v08d19 preload: {len(DECISION_ROWS_DF)} decisions from {_pfx}')\n"
    "else:\n"
    "    print('v08d19 preload: v08d18 cache not found, mining from scratch')\n\n"
    "if RUN_REPLAY_MINING:"
)
assert OLD_NO_PRELOAD in src7, 'no-preload block not found in Cell[7]'
src7 = src7.replace(OLD_NO_PRELOAD, NEW_PRELOAD)

cells[7]['source'] = src7.splitlines(keepends=True)

# ── Cell[19]: Add sequential opponent features ────────────────────────────────
src19 = ''.join(cells[19]['source'])

# 1. Add op_last_context to categorical features
OLD_CAT = "UNKNOWN0_CATEGORICAL_FEATURES = ['context_name', 'option_type', 'area', 'in_play_area', 'option_signature', 'opponent_archetype_norm']  # v08d10: add archetype"
NEW_CAT = "UNKNOWN0_CATEGORICAL_FEATURES = ['context_name', 'option_type', 'area', 'in_play_area', 'option_signature', 'opponent_archetype_norm', 'op_last_context']  # v08d10+v08d19"
assert OLD_CAT in src19, 'categorical features not found'
src19 = src19.replace(OLD_CAT, NEW_CAT)

# 2. Add sequential opponent feature computation after prize_gap block
OLD_EXTRA = (
    "        # Extra features from ALAKAZAM_OPTION_MODEL_DF\n"
    "        _extra_numeric_candidates = ['supporter_played', 'stadium_played', 'energy_attached',\n"
    "                                      'my_discard_count', 'op_discard_count', 'prize_gap']"
)
NEW_EXTRA = (
    "        # v08d19: sequential opponent features from DECISION_ROWS_DF step ordering\n"
    "        if 'DECISION_ROWS_DF' in globals() and DECISION_ROWS_DF is not None and not DECISION_ROWS_DF.empty:\n"
    "            _dr_seq = DECISION_ROWS_DF[['decision_id','episode_id','step','player_index','context_name']].copy()\n"
    "            _dr_seq = _dr_seq.sort_values(['episode_id','step']).reset_index(drop=True)\n"
    "            # UNKNOWN_0 Alakazam decisions\n"
    "            _unk0 = _dr_seq[(_dr_seq['context_name']=='UNKNOWN_0') &\n"
    "                            (_dr_seq['episode_id'].isin(work['episode_id'].unique()))].copy()\n"
    "            _op_parts = []\n"
    "            for (_ep, _pi), _grp in _unk0.groupby(['episode_id','player_index']):\n"
    "                _op_pi = 1 - _pi\n"
    "                _op_dr = _dr_seq[(_dr_seq['episode_id']==_ep) &\n"
    "                                  (_dr_seq['player_index']==_op_pi)].sort_values('step')\n"
    "                if len(_op_dr) == 0:\n"
    "                    _grp2 = _grp[['decision_id','step']].copy()\n"
    "                    _grp2['op_last_context'] = 'NONE'\n"
    "                    _grp2['steps_since_op'] = -1.0\n"
    "                    _op_parts.append(_grp2[['decision_id','op_last_context','steps_since_op']])\n"
    "                    continue\n"
    "                _m = pd.merge_asof(\n"
    "                    _grp[['decision_id','step']].sort_values('step'),\n"
    "                    _op_dr[['step','context_name']].rename(columns={'step':'op_step','context_name':'op_last_context'}),\n"
    "                    left_on='step', right_on='op_step', direction='backward'\n"
    "                )\n"
    "                _m['steps_since_op'] = (_m['step'] - _m.get('op_step', _m['step'])).fillna(-1.0)\n"
    "                _m['op_last_context'] = _m['op_last_context'].fillna('NONE')\n"
    "                _op_parts.append(_m[['decision_id','op_last_context','steps_since_op']])\n"
    "            if _op_parts:\n"
    "                _op_seq_df = pd.concat(_op_parts, ignore_index=True)\n"
    "                work = work.merge(_op_seq_df, on='decision_id', how='left')\n"
    "                work['op_last_context'] = work['op_last_context'].fillna('NONE')\n"
    "                work['steps_since_op'] = work['steps_since_op'].fillna(-1.0)\n"
    "                print(f'v08d19 op_last_context dist: {work[\"op_last_context\"].value_counts().head(8).to_dict()}')\n"
    "                print(f'v08d19 steps_since_op: mean={work[\"steps_since_op\"].mean():.1f}')\n"
    "            else:\n"
    "                work['op_last_context'] = 'NONE'\n"
    "                work['steps_since_op'] = -1.0\n"
    "        else:\n"
    "            work['op_last_context'] = 'NONE'\n"
    "            work['steps_since_op'] = -1.0\n"
    "        # Extra features from ALAKAZAM_OPTION_MODEL_DF\n"
    "        _extra_numeric_candidates = ['supporter_played', 'stadium_played', 'energy_attached',\n"
    "                                      'my_discard_count', 'op_discard_count', 'prize_gap', 'steps_since_op']"
)
assert OLD_EXTRA in src19, 'extra_numeric_candidates not found in Cell[19]'
src19 = src19.replace(OLD_EXTRA, NEW_EXTRA)

# 3. Add op_last_context proxy to injection code (before rows = [])
# At inference: approximate op_last_context from observable state
# op_active_energy_count=0 → opponent likely in ATTACH_FROM or early game
# op_active_energy_count>=2 → opponent likely attacking (UNKNOWN_0/MAIN)
OLD_INJ_PRIZES = (
    "        # v08d18: prize_gap — len(prize list) counts face-down prize cards correctly\\n"
    "        _my_prizes_left = float(len(list(getattr(my_ps, \\'prize\\', None) or [])))\\n"
    "        _op_prizes_left = float(len(list(getattr(op_ps, \\'prize\\', None) or [])))\\n"
    "        _prize_gap = _my_prizes_left - _op_prizes_left\\n"
    "        rows = []\\n        for i, o in enumerate(select.option):"
)
NEW_INJ_PRIZES = (
    "        # v08d18: prize_gap — len(prize list) counts face-down prize cards correctly\\n"
    "        _my_prizes_left = float(len(list(getattr(my_ps, \\'prize\\', None) or [])))\\n"
    "        _op_prizes_left = float(len(list(getattr(op_ps, \\'prize\\', None) or [])))\\n"
    "        _prize_gap = _my_prizes_left - _op_prizes_left\\n"
    "        # v08d19: op_last_context proxy from observable opponent state\\n"
    "        # exact context not observable; approximate from op energy count\\n"
    "        _op_energy = float(getattr(op_active, \\'energyCount\\', getattr(op_active, \\'energy_count\\', 0)) or 0)\\n"
    "        _op_last_ctx = \\'UNKNOWN_0\\' if _op_energy >= 2 else (\\'ATTACH_FROM\\' if _op_energy == 1 else \\'NONE\\')\\n"
    "        _steps_since_op = 2.0  # v08d19: approx (alternating turns)\\n"
    "        rows = []\\n        for i, o in enumerate(select.option):"
)
assert OLD_INJ_PRIZES in src19, 'injection prizes block not found in Cell[19]'
src19 = src19.replace(OLD_INJ_PRIZES, NEW_INJ_PRIZES)

# 4. Add op_last_context and steps_since_op to each row dict
OLD_INJ_END = (
    "                \\'opponent_archetype_norm\\': _op_arch,\\n"
    "                \\'prize_gap\\': _prize_gap,\\n"
    "            })"
)
NEW_INJ_END = (
    "                \\'opponent_archetype_norm\\': _op_arch,\\n"
    "                \\'prize_gap\\': _prize_gap,\\n"
    "                \\'op_last_context\\': _op_last_ctx,\\n"
    "                \\'steps_since_op\\': _steps_since_op,\\n"
    "            })"
)
assert OLD_INJ_END in src19, 'injection row dict end not found in Cell[19]'
src19 = src19.replace(OLD_INJ_END, NEW_INJ_END)

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

assert 'v0_08d19_op_seq' in s1,                         'EXPERIMENT_NAME'
assert "RUN_PREFIX = 'pokemon-20260627-v0-08d19'" in s1, 'RUN_PREFIX'
assert 'v08d18' in s7,                                   'preload from v08d18'
assert '_PRELOAD_CANDIDATES' in s7,                      'preload block'
assert 'op_last_context' in s19,                         'op_last_context feature'
assert 'steps_since_op' in s19,                          'steps_since_op feature'
assert 'merge_asof' in s19,                              'merge_asof sequential join'
assert '_op_last_ctx' in s19,                            'inference proxy'
assert 'prize_gap' in s19,                               'prize_gap kept'
assert 'opponent_archetype_norm' in s19,                 'opp_arch kept'
full = ''.join(''.join(c['source']) for c in nb2['cells'])
assert 'lambdarank' in full, 'lambdarank kept'
print('All sanity checks passed.')
