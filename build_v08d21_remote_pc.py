"""Build pokemon-20260627-v0-08d21-remote-pc.ipynb from v08d19.

v08d21: add my_prev_context — self-context before current UNKNOWN_0 decision.

Parallel to op_last_context (which tells us what OPPONENT was doing),
my_prev_context tells us what WE were doing just before this UNKNOWN_0 decision.

Training: merge_asof on same player, find most recent decision at step < T.
Inference: approximate from state_summary features:
  - energy_attached == 1 → 'ATTACH_FROM'
  - supporter_played == 1 → 'MAIN'
  - otherwise → 'NONE'

Hyperparams: same as v08d19 (63 leaves, 350k rows) to isolate feature effect.
Preload: from v08d19 (best top1=0.5469, same split).

Changes from v08d19:
  1. EXPERIMENT_NAME → v0_08d21_my_ctx
  2. RUN_PREFIX → pokemon-20260627-v0-08d21
  3. Cell[7]: Preload from v08d19
  4. Cell[19]: Add my_prev_context via merge_asof on same player
  5. Cell[19]: Add my_prev_context to categorical features
  6. Cell[19]: Revert num_leaves to 63 (was 63 in v08d19)
  7. Cell[1]: Revert MAX_CONTEXT_TRAIN_ROWS to 350_000 (was 350k in v08d19)
  8. Cell[19]: Injection: proxy my_prev_context from energy_attached/supporter_played
"""
import json

SRC = '/kaggle/working/pokemon-20260627-v0-08d20.ipynb'
DST = '/kaggle/working/pokemon-20260627-v0-08d21.ipynb'

with open(SRC) as f:
    nb = json.load(f)

cells = nb['cells']

# ── Cell[1]: Config ───────────────────────────────────────────────────────────
src1 = ''.join(cells[1]['source'])

OLD_EXP = "EXPERIMENT_NAME = 'v0_08d20_scale_up'"
NEW_EXP = "EXPERIMENT_NAME = 'v0_08d21_my_ctx'"
assert OLD_EXP in src1
src1 = src1.replace(OLD_EXP, NEW_EXP)

OLD_PFX = "RUN_PREFIX = 'pokemon-20260627-v0-08d20'"
NEW_PFX = "RUN_PREFIX = 'pokemon-20260627-v0-08d21'"
assert OLD_PFX in src1
src1 = src1.replace(OLD_PFX, NEW_PFX)

# Revert to 350k rows
OLD_MAX = "    MAX_CONTEXT_TRAIN_ROWS = 700_000\n    MAX_GLOBAL_TRAIN_ROWS = 1_400_000"
NEW_MAX = "    MAX_CONTEXT_TRAIN_ROWS = 350_000\n    MAX_GLOBAL_TRAIN_ROWS = 700_000"
assert OLD_MAX in src1, f'MAX_CONTEXT_TRAIN_ROWS not found: {repr(OLD_MAX[:60])}'
src1 = src1.replace(OLD_MAX, NEW_MAX)

cells[1]['source'] = src1.splitlines(keepends=True)

# ── Cell[7]: Preload from v08d19 ──────────────────────────────────────────────
src7 = ''.join(cells[7]['source'])

OLD_PRELOAD = (
    "# v08d20: preload from v08d19\n"
    "_PRELOAD_CANDIDATES = [\n"
    "    Path('/kaggle/working/pokemon-20260627-v0-08d19/pokemon-20260627-v0-08d19'),\n"
    "    Path('/kaggle/working/pokemon-20260627-v0-08d18/pokemon-20260627-v0-08d18'),\n"
    "]\n"
)
NEW_PRELOAD = (
    "# v08d21: preload from v08d19 (same split, 63 leaves, for fair comparison)\n"
    "_PRELOAD_CANDIDATES = [\n"
    "    Path('/kaggle/working/pokemon-20260627-v0-08d19/pokemon-20260627-v0-08d19'),\n"
    "    Path('/kaggle/working/pokemon-20260627-v0-08d18/pokemon-20260627-v0-08d18'),\n"
    "]\n"
)
assert OLD_PRELOAD in src7
src7 = src7.replace(OLD_PRELOAD, NEW_PRELOAD)

OLD_PRINT = "    print(f'v08d20 preload: {len(DECISION_ROWS_DF)} decisions from {_pfx}')\n"
NEW_PRINT = "    print(f'v08d21 preload: {len(DECISION_ROWS_DF)} decisions from {_pfx}')\n"
assert OLD_PRINT in src7
src7 = src7.replace(OLD_PRINT, NEW_PRINT)

OLD_FAIL = "    print('v08d20 preload: v08d19/v08d18 cache not found, mining from scratch')\n"
NEW_FAIL = "    print('v08d21 preload: v08d19/v08d18 cache not found, mining from scratch')\n"
assert OLD_FAIL in src7
src7 = src7.replace(OLD_FAIL, NEW_FAIL)

cells[7]['source'] = src7.splitlines(keepends=True)

# ── Cell[19]: Add my_prev_context ─────────────────────────────────────────────
src19 = ''.join(cells[19]['source'])

# 1. Revert num_leaves to 63
OLD_LEAVES = "'num_leaves': 127,"
NEW_LEAVES = "'num_leaves': 63,"
assert OLD_LEAVES in src19
src19 = src19.replace(OLD_LEAVES, NEW_LEAVES)

# 2. Add my_prev_context to categorical features (alongside op_last_context)
OLD_CAT = "UNKNOWN0_CATEGORICAL_FEATURES = ['context_name', 'option_type', 'area', 'in_play_area', 'option_signature', 'opponent_archetype_norm', 'op_last_context']  # v08d10+v08d19"
NEW_CAT = "UNKNOWN0_CATEGORICAL_FEATURES = ['context_name', 'option_type', 'area', 'in_play_area', 'option_signature', 'opponent_archetype_norm', 'op_last_context', 'my_prev_context']  # v08d10+v08d19+v08d21"
assert OLD_CAT in src19
src19 = src19.replace(OLD_CAT, NEW_CAT)

# 3. Extend the op_last_context computation block to also compute my_prev_context
# Find the block that ends with: _op_seq_df = concat and merge into work
OLD_SEQ_BLOCK = (
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
)
NEW_SEQ_BLOCK = (
    "        if 'DECISION_ROWS_DF' in globals() and DECISION_ROWS_DF is not None and not DECISION_ROWS_DF.empty:\n"
    "            _dr_seq = DECISION_ROWS_DF[['decision_id','episode_id','step','player_index','context_name']].copy()\n"
    "            _dr_seq = _dr_seq.sort_values(['episode_id','step']).reset_index(drop=True)\n"
    "            # UNKNOWN_0 Alakazam decisions\n"
    "            _unk0 = _dr_seq[(_dr_seq['context_name']=='UNKNOWN_0') &\n"
    "                            (_dr_seq['episode_id'].isin(work['episode_id'].unique()))].copy()\n"
    "            _op_parts = []\n"
    "            _my_parts = []\n"
    "            for (_ep, _pi), _grp in _unk0.groupby(['episode_id','player_index']):\n"
    "                _op_pi = 1 - _pi\n"
    "                _op_dr = _dr_seq[(_dr_seq['episode_id']==_ep) &\n"
    "                                  (_dr_seq['player_index']==_op_pi)].sort_values('step')\n"
    "                _my_dr = _dr_seq[(_dr_seq['episode_id']==_ep) &\n"
    "                                  (_dr_seq['player_index']==_pi)].sort_values('step')\n"
    "                _grp_s = _grp[['decision_id','step']].sort_values('step')\n"
    "                # op_last_context\n"
    "                if len(_op_dr) == 0:\n"
    "                    _grp2 = _grp_s.copy()\n"
    "                    _grp2['op_last_context'] = 'NONE'\n"
    "                    _grp2['steps_since_op'] = -1.0\n"
    "                    _op_parts.append(_grp2[['decision_id','op_last_context','steps_since_op']])\n"
    "                else:\n"
    "                    _m = pd.merge_asof(\n"
    "                        _grp_s,\n"
    "                        _op_dr[['step','context_name']].rename(columns={'step':'op_step','context_name':'op_last_context'}),\n"
    "                        left_on='step', right_on='op_step', direction='backward'\n"
    "                    )\n"
    "                    _m['steps_since_op'] = (_m['step'] - _m.get('op_step', _m['step'])).fillna(-1.0)\n"
    "                    _m['op_last_context'] = _m['op_last_context'].fillna('NONE')\n"
    "                    _op_parts.append(_m[['decision_id','op_last_context','steps_since_op']])\n"
    "                # my_prev_context: same player's most recent decision strictly before current step\n"
    "                if len(_my_dr) < 2:\n"
    "                    _grp3 = _grp_s.copy()\n"
    "                    _grp3['my_prev_context'] = 'NONE'\n"
    "                    _my_parts.append(_grp3[['decision_id','my_prev_context']])\n"
    "                else:\n"
    "                    _m2 = pd.merge_asof(\n"
    "                        _grp_s,\n"
    "                        _my_dr[['step','context_name']].rename(columns={'step':'my_step','context_name':'my_prev_context'}),\n"
    "                        left_on='step', right_on='my_step', direction='backward'\n"
    "                    )\n"
    "                    # Exclude the current decision itself (step == my_step)\n"
    "                    _m2.loc[_m2['step'] == _m2.get('my_step', _m2['step']), 'my_prev_context'] = 'NONE'\n"
    "                    _m2['my_prev_context'] = _m2['my_prev_context'].fillna('NONE')\n"
    "                    _my_parts.append(_m2[['decision_id','my_prev_context']])\n"
    "            if _op_parts:\n"
    "                _op_seq_df = pd.concat(_op_parts, ignore_index=True)\n"
    "                work = work.merge(_op_seq_df, on='decision_id', how='left')\n"
    "                work['op_last_context'] = work['op_last_context'].fillna('NONE')\n"
    "                work['steps_since_op'] = work['steps_since_op'].fillna(-1.0)\n"
    "                print(f'v08d21 op_last_context dist: {work[\"op_last_context\"].value_counts().head(6).to_dict()}')\n"
    "            else:\n"
    "                work['op_last_context'] = 'NONE'\n"
    "                work['steps_since_op'] = -1.0\n"
    "            if _my_parts:\n"
    "                _my_seq_df = pd.concat(_my_parts, ignore_index=True)\n"
    "                work = work.merge(_my_seq_df, on='decision_id', how='left')\n"
    "                work['my_prev_context'] = work['my_prev_context'].fillna('NONE')\n"
    "                print(f'v08d21 my_prev_context dist: {work[\"my_prev_context\"].value_counts().head(6).to_dict()}')\n"
    "            else:\n"
    "                work['my_prev_context'] = 'NONE'\n"
    "        else:\n"
    "            work['op_last_context'] = 'NONE'\n"
    "            work['steps_since_op'] = -1.0\n"
    "            work['my_prev_context'] = 'NONE'\n"
)
assert OLD_SEQ_BLOCK in src19, 'sequential block not found'
src19 = src19.replace(OLD_SEQ_BLOCK, NEW_SEQ_BLOCK)

# 4. Add my_prev_context inference proxy (before rows = [])
OLD_INJ_CTX = (
    "        # v08d19: op_last_context proxy from observable opponent state\\n"
    "        # exact context not observable; approximate from op energy count\\n"
    "        _op_energy = float(getattr(op_active, \\'energyCount\\', getattr(op_active, \\'energy_count\\', 0)) or 0)\\n"
    "        _op_last_ctx = \\'UNKNOWN_0\\' if _op_energy >= 2 else (\\'ATTACH_FROM\\' if _op_energy == 1 else \\'NONE\\')\\n"
    "        rows = []\\n        for i, o in enumerate(select.option):"
)
NEW_INJ_CTX = (
    "        # v08d19: op_last_context proxy from observable opponent state\\n"
    "        _op_energy = float(getattr(op_active, \\'energyCount\\', getattr(op_active, \\'energy_count\\', 0)) or 0)\\n"
    "        _op_last_ctx = \\'UNKNOWN_0\\' if _op_energy >= 2 else (\\'ATTACH_FROM\\' if _op_energy == 1 else \\'NONE\\')\\n"
    "        # v08d21: my_prev_context proxy from state_summary flags\\n"
    "        _my_energy_att = float(getattr(my_ps, \\'energyAttached\\', getattr(my_ps, \\'energy_attached\\', 0)) or 0)\\n"
    "        _my_sup_played = float(getattr(my_ps, \\'supporterPlayed\\', getattr(my_ps, \\'supporter_played\\', 0)) or 0)\\n"
    "        _my_prev_ctx = \\'ATTACH_FROM\\' if _my_energy_att > 0 else (\\'MAIN\\' if _my_sup_played > 0 else \\'NONE\\')\\n"
    "        rows = []\\n        for i, o in enumerate(select.option):"
)
assert OLD_INJ_CTX in src19, 'injection context block not found'
src19 = src19.replace(OLD_INJ_CTX, NEW_INJ_CTX)

# 5. Add my_prev_context to row dict
OLD_INJ_END = (
    "                \\'op_last_context\\': _op_last_ctx,\\n"
    "            })"
)
NEW_INJ_END = (
    "                \\'op_last_context\\': _op_last_ctx,\\n"
    "                \\'my_prev_context\\': _my_prev_ctx,\\n"
    "            })"
)
assert OLD_INJ_END in src19, 'injection row dict end not found'
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

assert 'v0_08d21_my_ctx' in s1
assert "RUN_PREFIX = 'pokemon-20260627-v0-08d21'" in s1
assert 'MAX_CONTEXT_TRAIN_ROWS = 350_000' in s1, 'reverted to 350k'
assert 'v08d19' in s7, 'preload from v08d19'
assert "'num_leaves': 63" in s19, 'reverted to 63 leaves'
assert 'my_prev_context' in s19, 'my_prev_context feature'
assert '_my_prev_ctx' in s19, 'inference proxy'
assert 'op_last_context' in s19, 'op_last_context kept'
assert 'prize_gap' in s19, 'prize_gap kept'
full = ''.join(''.join(c['source']) for c in nb2['cells'])
assert 'lambdarank' in full
print('All sanity checks passed.')
