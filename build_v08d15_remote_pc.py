"""Build pokemon-20260627-v0-08d15-remote-pc.ipynb from v08d2.

v08d15: filter training to Alakazam-player decisions only.

KEY INSIGHT: The LGBM model should mimic what ALAKAZAM EXPERTS do, not
a mixture of all deck archetypes. Currently:
  - Alakazam decisions: 1.32M rows (24.6% of UNKNOWN_0)
  - Non-Alakazam decisions: 4.1M rows (75.4%) — NOISE for our agent

Our Alakazam agent:
  - Has Alakazam deck composition
  - Makes energy attachment / attack / retreat decisions for Alakazam
  - Should mimic Alakazam expert behavior ONLY

Filtering to player_archetype='Alakazam' rows:
  - Cleaner training signal for our exact agent context
  - 1.32M rows still sufficient for LambdaRank
  - Hypothesis: alignment of training decision-maker with inference agent gives breakthrough

Also includes opponent_archetype_norm (from v08d10, +0.0015).

Changes from v08d2:
  1. EXPERIMENT_NAME → v0_08d15_alakazam_filter
  2. RUN_PREFIX → pokemon-20260627-v0-08d15
  3. Cell[19]: filter to Alakazam player decisions + join opponent_archetype
  4. Cell[7]: Preload from v08d1 cache
"""
import json

SRC = '/kaggle/working/pokemon-20260627-v0-08d2.ipynb'
DST = '/kaggle/working/pokemon-20260627-v0-08d15.ipynb'

with open(SRC) as f:
    nb = json.load(f)

cells = nb['cells']

# ── Cell[1]: Config ───────────────────────────────────────────────────────────
src1 = ''.join(cells[1]['source'])

OLD_EXP = "EXPERIMENT_NAME = 'v0_08d2_top200_lgbm_noeval'"
NEW_EXP = "EXPERIMENT_NAME = 'v0_08d15_alakazam_filter'"
assert OLD_EXP in src1
src1 = src1.replace(OLD_EXP, NEW_EXP)

OLD_PFX = "RUN_PREFIX = 'pokemon-20260627-v0-08d2'"
NEW_PFX = "RUN_PREFIX = 'pokemon-20260627-v0-08d15'"
assert OLD_PFX in src1
src1 = src1.replace(OLD_PFX, NEW_PFX)

cells[1]['source'] = src1.splitlines(keepends=True)

# ── Cell[19]: Alakazam-filter + opponent_archetype ────────────────────────────
src19 = ''.join(cells[19]['source'])

# 1. Add opponent_archetype to categorical features (from v08d10)
OLD_CAT = "UNKNOWN0_CATEGORICAL_FEATURES = ['context_name', 'option_type', 'area', 'in_play_area', 'option_signature']"
NEW_CAT = "UNKNOWN0_CATEGORICAL_FEATURES = ['context_name', 'option_type', 'area', 'in_play_area', 'option_signature', 'opponent_archetype_norm']  # v08d15: add archetype"
assert OLD_CAT in src19
src19 = src19.replace(OLD_CAT, NEW_CAT)

# 2. Filter to Alakazam-player decisions + join opponent_archetype
OLD_WORK = (
    "        work = prepare_unknown0_training_frame(ALAKAZAM_OPTION_MODEL_DF, max_rows=max_rows)\n"
    "        # Extra features from ALAKAZAM_OPTION_MODEL_DF\n"
    "        _extra_numeric_candidates = ['supporter_played', 'stadium_played', 'energy_attached',\n"
    "                                      'my_discard_count', 'op_discard_count']"
)
NEW_WORK = (
    "        work = prepare_unknown0_training_frame(ALAKAZAM_OPTION_MODEL_DF, max_rows=max_rows)\n"
    "        # v08d15: join opponent_archetype + filter to Alakazam-player decisions\n"
    "        _v15_arch_map = {'Hop_Trevenant': 'hop_control', 'Alakazam': 'alakazam_mirror',\n"
    "                         'Mega_Lucario': 'lucario', 'Other': 'generic_control'}\n"
    "        if 'DECISION_ROWS_DF' in globals() and DECISION_ROWS_DF is not None:\n"
    "            _dec = DECISION_ROWS_DF[['decision_id', 'player_archetype', 'opponent_archetype']].drop_duplicates('decision_id').copy()\n"
    "            _dec['player_archetype_norm'] = _dec['player_archetype'].map(_v15_arch_map).fillna('generic_control')\n"
    "            _dec['opponent_archetype_norm'] = _dec['opponent_archetype'].map(_v15_arch_map).fillna('generic_control')\n"
    "            _n_before = len(work)\n"
    "            work = work.merge(_dec[['decision_id', 'player_archetype_norm', 'opponent_archetype_norm']], on='decision_id', how='inner')\n"
    "            # Filter to Alakazam-player decisions ONLY\n"
    "            work = work[work['player_archetype_norm'] == 'alakazam_mirror'].copy()\n"
    "            work['opponent_archetype_norm'] = work['opponent_archetype_norm'].fillna('generic_control')\n"
    "            print(f'v08d15 Alakazam filter: {len(work)}/{_n_before} rows ({100*len(work)/_n_before:.1f}%)')\n"
    "            print(f'  opp_arch dist: {work[\"opponent_archetype_norm\"].value_counts().to_dict()}')\n"
    "        else:\n"
    "            work['opponent_archetype_norm'] = 'generic_control'\n"
    "        # Extra features from ALAKAZAM_OPTION_MODEL_DF\n"
    "        _extra_numeric_candidates = ['supporter_played', 'stadium_played', 'energy_attached',\n"
    "                                      'my_discard_count', 'op_discard_count']"
)
assert OLD_WORK in src19
src19 = src19.replace(OLD_WORK, NEW_WORK)

# 3. Add detect_opponent_archetype to injection code (from v08d10)
OLD_INJ_ROWS = "rows = []\\n        for i, o in enumerate(select.option):"
NEW_INJ_ROWS = (
    "_op_all = ([op_active] if op_active else []) + op_bench\\n"
    "        _op_arch, _ = detect_opponent_archetype(_op_all, float(getattr(getattr(obs.current, \\'stadium\\', None), \\'id\\', 0) or 0))\\n"
    "        rows = []\\n        for i, o in enumerate(select.option):"
)
assert OLD_INJ_ROWS in src19
src19 = src19.replace(OLD_INJ_ROWS, NEW_INJ_ROWS)

OLD_INJ_SIG = "   \\'option_signature\\': sig,\\n            })"
NEW_INJ_SIG = "   \\'option_signature\\': sig,\\n                \\'opponent_archetype_norm\\': _op_arch,\\n            })"
assert OLD_INJ_SIG in src19
src19 = src19.replace(OLD_INJ_SIG, NEW_INJ_SIG)

cells[19]['source'] = src19.splitlines(keepends=True)

# ── Cell[7]: Preload episode data ─────────────────────────────────────────────
src7 = ''.join(cells[7]['source'])

OLD_PRELOAD_POINT = 'NON_EPISODE_JSON_ROWS = []\n\nif RUN_REPLAY_MINING:'
NEW_PRELOAD_POINT = (
    "NON_EPISODE_JSON_ROWS = []\n\n"
    "# v08d15: preload from prior run cache\n"
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
    "    print(f'v08d15 preload: from {_pfx}')\n"
    "else:\n"
    "    print('v08d15 preload: no cache, mining from scratch')\n\n"
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

assert 'v0_08d15_alakazam_filter' in s1,            'EXPERIMENT_NAME'
assert "RUN_PREFIX = 'pokemon-20260627-v0-08d15'" in s1, 'RUN_PREFIX'
assert 'RUN_LOCAL_EVAL = False' in s1,              'eval disabled'
assert 'alakazam_mirror' in s19,                    'Alakazam filter'
assert 'player_archetype_norm' in s19,              'player_archetype_norm join'
assert 'opponent_archetype_norm' in s19,            'opp archetype feature'
assert '_op_arch' in s19,                           'archetype inference'
assert 'v08d15 preload' in s7,                      'preload logic'
full = ''.join(''.join(c['source']) for c in nb2['cells'])
assert 'lambdarank' in full, 'lambdarank kept'
print('All sanity checks passed.')
