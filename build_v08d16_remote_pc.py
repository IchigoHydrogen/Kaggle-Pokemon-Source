"""Build pokemon-20260627-v0-08d16-remote-pc.ipynb from v08d2.

v08d16: add player_archetype_norm as feature (keep all data, hardcode at inference).

Rather than filtering (v08d15), add the player's deck archetype as an EXPLICIT
feature so LGBM can learn archetype-specific behaviors while using all data:
  - Training: player_archetype_norm ∈ {hop_control, alakazam_mirror, lucario, generic_control}
  - Inference: player_archetype_norm = 'alakazam_mirror' (hardcoded — we ARE Alakazam)

Combined with opponent_archetype_norm (v08d10), this gives LGBM the full matchup:
  "player=alakazam_mirror vs opponent=hop_control → specific strategy learned"

The model at inference effectively becomes: "I'm Alakazam, what does an Alakazam
expert do vs THIS opponent archetype?"

Also includes opponent_archetype_norm (from v08d10, +0.0015).

Changes from v08d2:
  1. EXPERIMENT_NAME → v0_08d16_player_arch
  2. RUN_PREFIX → pokemon-20260627-v0-08d16
  3. Cell[19]: Add player_archetype_norm + opponent_archetype_norm as features
  4. Cell[19]: _LGBM_INJ_CODE: hardcode player_archetype_norm='alakazam_mirror'
  5. Cell[7]: Preload from v08d1 cache
"""
import json

SRC = '/kaggle/working/pokemon-20260627-v0-08d2.ipynb'
DST = '/kaggle/working/pokemon-20260627-v0-08d16.ipynb'

with open(SRC) as f:
    nb = json.load(f)

cells = nb['cells']

# ── Cell[1]: Config ───────────────────────────────────────────────────────────
src1 = ''.join(cells[1]['source'])

OLD_EXP = "EXPERIMENT_NAME = 'v0_08d2_top200_lgbm_noeval'"
NEW_EXP = "EXPERIMENT_NAME = 'v0_08d16_player_arch'"
assert OLD_EXP in src1
src1 = src1.replace(OLD_EXP, NEW_EXP)

OLD_PFX = "RUN_PREFIX = 'pokemon-20260627-v0-08d2'"
NEW_PFX = "RUN_PREFIX = 'pokemon-20260627-v0-08d16'"
assert OLD_PFX in src1
src1 = src1.replace(OLD_PFX, NEW_PFX)

cells[1]['source'] = src1.splitlines(keepends=True)

# ── Cell[19]: Both archetypes as features ─────────────────────────────────────
src19 = ''.join(cells[19]['source'])

# 1. Add BOTH player + opponent archetypes to categorical features
OLD_CAT = "UNKNOWN0_CATEGORICAL_FEATURES = ['context_name', 'option_type', 'area', 'in_play_area', 'option_signature']"
NEW_CAT = "UNKNOWN0_CATEGORICAL_FEATURES = ['context_name', 'option_type', 'area', 'in_play_area', 'option_signature', 'player_archetype_norm', 'opponent_archetype_norm']  # v08d16: both archetypes"
assert OLD_CAT in src19
src19 = src19.replace(OLD_CAT, NEW_CAT)

# 2. Join both archetypes from DECISION_ROWS_DF
OLD_WORK = (
    "        work = prepare_unknown0_training_frame(ALAKAZAM_OPTION_MODEL_DF, max_rows=max_rows)\n"
    "        # Extra features from ALAKAZAM_OPTION_MODEL_DF\n"
    "        _extra_numeric_candidates = ['supporter_played', 'stadium_played', 'energy_attached',\n"
    "                                      'my_discard_count', 'op_discard_count']"
)
NEW_WORK = (
    "        work = prepare_unknown0_training_frame(ALAKAZAM_OPTION_MODEL_DF, max_rows=max_rows)\n"
    "        # v08d16: join player_archetype_norm + opponent_archetype_norm\n"
    "        _v16_arch_map = {'Hop_Trevenant': 'hop_control', 'Alakazam': 'alakazam_mirror',\n"
    "                         'Mega_Lucario': 'lucario', 'Other': 'generic_control'}\n"
    "        if 'DECISION_ROWS_DF' in globals() and DECISION_ROWS_DF is not None:\n"
    "            _dec = DECISION_ROWS_DF[['decision_id', 'player_archetype', 'opponent_archetype']].drop_duplicates('decision_id').copy()\n"
    "            _dec['player_archetype_norm'] = _dec['player_archetype'].map(_v16_arch_map).fillna('generic_control')\n"
    "            _dec['opponent_archetype_norm'] = _dec['opponent_archetype'].map(_v16_arch_map).fillna('generic_control')\n"
    "            work = work.merge(_dec[['decision_id', 'player_archetype_norm', 'opponent_archetype_norm']], on='decision_id', how='left')\n"
    "            work['player_archetype_norm'] = work['player_archetype_norm'].fillna('generic_control')\n"
    "            work['opponent_archetype_norm'] = work['opponent_archetype_norm'].fillna('generic_control')\n"
    "            print(f'v08d16 player_arch dist: {work[\"player_archetype_norm\"].value_counts().to_dict()}')\n"
    "            print(f'v08d16 opp_arch dist:    {work[\"opponent_archetype_norm\"].value_counts().to_dict()}')\n"
    "        else:\n"
    "            work['player_archetype_norm'] = 'generic_control'\n"
    "            work['opponent_archetype_norm'] = 'generic_control'\n"
    "        # Extra features from ALAKAZAM_OPTION_MODEL_DF\n"
    "        _extra_numeric_candidates = ['supporter_played', 'stadium_played', 'energy_attached',\n"
    "                                      'my_discard_count', 'op_discard_count']"
)
assert OLD_WORK in src19
src19 = src19.replace(OLD_WORK, NEW_WORK)

# 3. Add detect_opponent_archetype to injection code + hardcode player_archetype
OLD_INJ_ROWS = "rows = []\\n        for i, o in enumerate(select.option):"
NEW_INJ_ROWS = (
    "_op_all = ([op_active] if op_active else []) + op_bench\\n"
    "        _op_arch, _ = detect_opponent_archetype(_op_all, float(getattr(getattr(obs.current, \\'stadium\\', None), \\'id\\', 0) or 0))\\n"
    "        _my_arch = \\'alakazam_mirror\\'  # v08d16: hardcoded — we are always Alakazam\\n"
    "        rows = []\\n        for i, o in enumerate(select.option):"
)
assert OLD_INJ_ROWS in src19
src19 = src19.replace(OLD_INJ_ROWS, NEW_INJ_ROWS)

# 4. Add both archetypes to each row
OLD_INJ_SIG = "   \\'option_signature\\': sig,\\n            })"
NEW_INJ_SIG = (
    "   \\'option_signature\\': sig,\\n"
    "                \\'player_archetype_norm\\': _my_arch,\\n"
    "                \\'opponent_archetype_norm\\': _op_arch,\\n"
    "            })"
)
assert OLD_INJ_SIG in src19
src19 = src19.replace(OLD_INJ_SIG, NEW_INJ_SIG)

cells[19]['source'] = src19.splitlines(keepends=True)

# ── Cell[7]: Preload episode data ─────────────────────────────────────────────
src7 = ''.join(cells[7]['source'])

OLD_PRELOAD_POINT = 'NON_EPISODE_JSON_ROWS = []\n\nif RUN_REPLAY_MINING:'
NEW_PRELOAD_POINT = (
    "NON_EPISODE_JSON_ROWS = []\n\n"
    "# v08d16: preload from prior run cache\n"
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
    "    print(f'v08d16 preload: from {_pfx}')\n"
    "else:\n"
    "    print('v08d16 preload: no cache, mining from scratch')\n\n"
    "if RUN_REPLAY_MINING:"
)
assert OLD_PRELOAD_POINT in src7
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

assert 'v0_08d16_player_arch' in s1,               'EXPERIMENT_NAME'
assert "RUN_PREFIX = 'pokemon-20260627-v0-08d16'" in s1, 'RUN_PREFIX'
assert 'RUN_LOCAL_EVAL = False' in s1,             'eval disabled'
assert 'player_archetype_norm' in s19,             'player_archetype_norm feature'
assert 'opponent_archetype_norm' in s19,           'opponent_archetype_norm feature'
assert '_my_arch' in s19,                          'player arch at inference'
assert '_op_arch' in s19,                          'opp arch at inference'
assert "alakazam_mirror" in s19,                   'hardcoded Alakazam'
assert 'v08d16 preload' in s7,                     'preload logic'
full = ''.join(''.join(c['source']) for c in nb2['cells'])
assert 'lambdarank' in full, 'lambdarank kept'
print('All sanity checks passed.')
