"""Build pokemon-20260627-v0-08d10-remote-pc.ipynb from v08d2.

v08d10: add opponent_archetype_norm as LGBM feature.

Currently the UNKNOWN_0 LGBM only knows board state (HP, energy, card counts)
but not WHO the opponent is playing. Adding opponent archetype lets the model
learn context-specific decisions:
- vs hop_control: different optimal plays than vs lucario
- vs alakazam_mirror: specific counter-strategies

The Alakazam agent already detects opponent archetype via detect_opponent_archetype().
We map its 4 return values to match the training data labels:
  'hop_control' = Hop_Trevenant deck (24.8% of games)
  'alakazam_mirror' = Alakazam deck (19.5% of games)
  'lucario' = Mega_Lucario deck (16.9% of games)
  'generic_control' = Other (38.8% of games)

Changes from v08d2:
  1. EXPERIMENT_NAME → v0_08d10_opp_arch
  2. RUN_PREFIX → pokemon-20260627-v0-08d10
  3. Cell[19]: After prepare_unknown0_training_frame, join opponent_archetype from DECISION_ROWS_DF
  4. Cell[19]: Add 'opponent_archetype_norm' to UNKNOWN0_CATEGORICAL_FEATURES
  5. Cell[19]: Inject detect_opponent_archetype call + feature in _LGBM_INJ_CODE
"""
import json

SRC = '/kaggle/working/pokemon-20260627-v0-08d2.ipynb'
DST = '/kaggle/working/pokemon-20260627-v0-08d10.ipynb'

with open(SRC) as f:
    nb = json.load(f)

cells = nb['cells']

# ── Cell[1]: Config ───────────────────────────────────────────────────────────
src1 = ''.join(cells[1]['source'])

OLD_EXP = "EXPERIMENT_NAME = 'v0_08d2_top200_lgbm_noeval'"
NEW_EXP = "EXPERIMENT_NAME = 'v0_08d10_opp_arch'"
assert OLD_EXP in src1
src1 = src1.replace(OLD_EXP, NEW_EXP)

OLD_PFX = "RUN_PREFIX = 'pokemon-20260627-v0-08d2'"
NEW_PFX = "RUN_PREFIX = 'pokemon-20260627-v0-08d10'"
assert OLD_PFX in src1
src1 = src1.replace(OLD_PFX, NEW_PFX)

cells[1]['source'] = src1.splitlines(keepends=True)

# ── Cell[19]: Add opponent_archetype feature ──────────────────────────────────
src19 = ''.join(cells[19]['source'])

# 1. Add to UNKNOWN0_CATEGORICAL_FEATURES
OLD_CAT = "UNKNOWN0_CATEGORICAL_FEATURES = ['context_name', 'option_type', 'area', 'in_play_area', 'option_signature']"
NEW_CAT = "UNKNOWN0_CATEGORICAL_FEATURES = ['context_name', 'option_type', 'area', 'in_play_area', 'option_signature', 'opponent_archetype_norm']  # v08d10: add archetype"
assert OLD_CAT in src19, f'CATEGORICAL_FEATURES not found'
src19 = src19.replace(OLD_CAT, NEW_CAT)

# 2. After 'work = prepare_unknown0_training_frame(...)' insert archetype join
OLD_WORK = (
    "        work = prepare_unknown0_training_frame(ALAKAZAM_OPTION_MODEL_DF, max_rows=max_rows)\n"
    "        # Extra features from ALAKAZAM_OPTION_MODEL_DF\n"
    "        _extra_numeric_candidates = ['supporter_played', 'stadium_played', 'energy_attached',\n"
    "                                      'my_discard_count', 'op_discard_count']"
)
NEW_WORK = (
    "        work = prepare_unknown0_training_frame(ALAKAZAM_OPTION_MODEL_DF, max_rows=max_rows)\n"
    "        # v08d10: join opponent_archetype from DECISION_ROWS_DF\n"
    "        _v10_arch_map = {'Hop_Trevenant': 'hop_control', 'Alakazam': 'alakazam_mirror',\n"
    "                         'Mega_Lucario': 'lucario', 'Other': 'generic_control'}\n"
    "        if 'DECISION_ROWS_DF' in globals() and DECISION_ROWS_DF is not None and 'opponent_archetype' in DECISION_ROWS_DF.columns:\n"
    "            _arch_df = DECISION_ROWS_DF[['decision_id', 'opponent_archetype']].drop_duplicates('decision_id').copy()\n"
    "            _arch_df['opponent_archetype_norm'] = _arch_df['opponent_archetype'].map(_v10_arch_map).fillna('generic_control')\n"
    "            work = work.merge(_arch_df[['decision_id', 'opponent_archetype_norm']], on='decision_id', how='left')\n"
    "            work['opponent_archetype_norm'] = work['opponent_archetype_norm'].fillna('generic_control')\n"
    "            print(f'v08d10 archetype distribution: {work[\"opponent_archetype_norm\"].value_counts().to_dict()}')\n"
    "        else:\n"
    "            work['opponent_archetype_norm'] = 'generic_control'\n"
    "            print('v08d10: DECISION_ROWS_DF missing opponent_archetype, using generic_control')\n"
    "        # Extra features from ALAKAZAM_OPTION_MODEL_DF\n"
    "        _extra_numeric_candidates = ['supporter_played', 'stadium_played', 'energy_attached',\n"
    "                                      'my_discard_count', 'op_discard_count']"
)
assert OLD_WORK in src19, 'prepare_unknown0_training_frame call not found'
src19 = src19.replace(OLD_WORK, NEW_WORK)

# 3. Modify _LGBM_INJ_CODE to add detect_opponent_archetype call + feature
# Add archetype detection before rows = []
OLD_INJ_ROWS = "rows = []\\n        for i, o in enumerate(select.option):"
NEW_INJ_ROWS = (
    "_op_all = ([op_active] if op_active else []) + op_bench\\n"
    "        _op_arch, _ = detect_opponent_archetype(_op_all, float(getattr(getattr(obs.current, \\'stadium\\', None), \\'id\\', 0) or 0))\\n"
    "        rows = []\\n        for i, o in enumerate(select.option):"
)
assert OLD_INJ_ROWS in src19, f'rows=[] in injection code not found'
src19 = src19.replace(OLD_INJ_ROWS, NEW_INJ_ROWS)

# Add opponent_archetype_norm to each row dict (after option_signature)
OLD_INJ_ARCH = "   \\'option_signature\\': sig,\\n            })"
NEW_INJ_ARCH = "   \\'option_signature\\': sig,\\n                \\'opponent_archetype_norm\\': _op_arch,\\n            })"
assert OLD_INJ_ARCH in src19, f'option_signature row dict closing not found'
src19 = src19.replace(OLD_INJ_ARCH, NEW_INJ_ARCH)

cells[19]['source'] = src19.splitlines(keepends=True)

# ── Cell[7]: Preload episode data ─────────────────────────────────────────────
src7 = ''.join(cells[7]['source'])

OLD_PRELOAD_POINT = 'NON_EPISODE_JSON_ROWS = []\n\nif RUN_REPLAY_MINING:'
NEW_PRELOAD_POINT = (
    "NON_EPISODE_JSON_ROWS = []\n\n"
    "# v08d10: preload pre-processed episode data from prior run cache\n"
    "_PRELOAD_CANDIDATES = [\n"
    "    Path('/kaggle/working/pokemon-20260627-v0-08d2/pokemon-20260627-v0-08d2'),\n"
    "    Path('/kaggle/working/pokemon-20260627-v0-08d1/pokemon-20260627-v0-08d1'),\n"
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
    "    print(f'v08d10 preload: {len(DECISION_ROWS_DF)} decisions, {len(OPTION_ROWS_DF)} options from {_pfx}')\n"
    "else:\n"
    "    print('v08d10 preload: no cache found, running episode mining from scratch')\n\n"
    "if RUN_REPLAY_MINING:"
)
assert OLD_PRELOAD_POINT in src7, f'preload patch point not found in Cell[7]'
src7 = src7.replace(OLD_PRELOAD_POINT, NEW_PRELOAD_POINT)

cells[7]['source'] = src7.splitlines(keepends=True)

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
s1 = ''.join(nb2['cells'][1]['source'])
s19 = ''.join(nb2['cells'][19]['source'])
s7 = ''.join(nb2['cells'][7]['source'])

assert 'v0_08d10_opp_arch' in s1,                   'EXPERIMENT_NAME'
assert "RUN_PREFIX = 'pokemon-20260627-v0-08d10'" in s1, 'RUN_PREFIX'
assert 'RUN_LOCAL_EVAL = False' in s1,               'eval disabled'
assert 'opponent_archetype_norm' in s19,             'archetype feature in Cell[19]'
assert '_v10_arch_map' in s19,                       'archetype mapping in Cell[19]'
assert '_op_arch' in s19,                            'archetype in injection code'
assert 'v08d10 preload' in s7,                       'preload logic in Cell[7]'
assert '_PRELOAD_CANDIDATES' in s7,                  'preload candidates in Cell[7]'
full = ''.join(''.join(c['source']) for c in nb2['cells'])
assert 'lambdarank' in full, 'lambdarank kept'
print('All sanity checks passed.')
