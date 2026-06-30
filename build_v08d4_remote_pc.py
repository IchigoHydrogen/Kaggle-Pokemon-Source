"""Build pokemon-20260627-v0-08d4-remote-pc.ipynb from v08d2.

v08d4: exclude option_index from LGBM features.

Finding from v08d1: option_index was #1 feature with 4x importance vs #2.
The model learned a positional heuristic (position 0 = first attack, etc).
This may be a valid signal but it might also be biased by game state correlations.

Hypothesis: removing option_index forces the LGBM to learn non-trivial
game state features (board state, energy counts, HP, etc.) that generalize
better to diverse opponent strategies. May hurt local top-1 accuracy
but improve actual game win rate.

Changes from v08d2:
  1. EXPERIMENT_NAME → v0_08d4_top200_lgbm_no_optidx
  2. RUN_PREFIX → pokemon-20260627-v0-08d4
  3. Cell[19]: UNKNOWN0_LGBM_EXCLUDE_FEATURES add 'option_index'
     OLD: UNKNOWN0_LGBM_EXCLUDE_FEATURES = ['turn_action_count']
     NEW: UNKNOWN0_LGBM_EXCLUDE_FEATURES = ['turn_action_count', 'option_index']
"""
import json

SRC = '/kaggle/working/pokemon-20260627-v0-08d2.ipynb'
DST = '/kaggle/working/pokemon-20260627-v0-08d4.ipynb'

with open(SRC) as f:
    nb = json.load(f)

cells = nb['cells']

# ── Cell[1]: Config ───────────────────────────────────────────────────────────
src1 = ''.join(cells[1]['source'])

OLD_EXP = "EXPERIMENT_NAME = 'v0_08d2_top200_lgbm_noeval'"
NEW_EXP = "EXPERIMENT_NAME = 'v0_08d4_top200_lgbm_no_optidx'"
assert OLD_EXP in src1, f'EXPERIMENT_NAME not found'
src1 = src1.replace(OLD_EXP, NEW_EXP)

OLD_PFX = "RUN_PREFIX = 'pokemon-20260627-v0-08d2'"
NEW_PFX = "RUN_PREFIX = 'pokemon-20260627-v0-08d4'"
assert OLD_PFX in src1, f'RUN_PREFIX not found'
src1 = src1.replace(OLD_PFX, NEW_PFX)

cells[1]['source'] = src1.splitlines(keepends=True)

# ── Cell[19]: Exclude option_index from features ──────────────────────────────
src19 = ''.join(cells[19]['source'])

OLD_EX = "UNKNOWN0_LGBM_EXCLUDE_FEATURES = ['turn_action_count']"
NEW_EX = "UNKNOWN0_LGBM_EXCLUDE_FEATURES = ['turn_action_count', 'option_index']   # v08d4: remove positional bias"
assert OLD_EX in src19, f'UNKNOWN0_LGBM_EXCLUDE_FEATURES not found'
src19 = src19.replace(OLD_EX, NEW_EX)

cells[19]['source'] = src19.splitlines(keepends=True)

# ── Cell[7]: Preload episode data from v08d2 cache ────────────────────────────
src7 = ''.join(cells[7]['source'])

OLD_PRELOAD_POINT = 'NON_EPISODE_JSON_ROWS = []\n\nif RUN_REPLAY_MINING:'
NEW_PRELOAD_POINT = (
    "NON_EPISODE_JSON_ROWS = []\n\n"
    "# v08d4+: preload pre-processed episode data from v08d2 cache\n"
    "# Try v08d2 first, fall back to v08d1 (same data, just different run)\n"
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
    "    print(f'v08d4 preload: {len(DECISION_ROWS_DF)} decisions, {len(OPTION_ROWS_DF)} options from {_pfx}')\n"
    "else:\n"
    "    print('v08d4 preload: no cache found, running episode mining from scratch')\n\n"
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

assert 'v0_08d4_top200_lgbm_no_optidx' in s1,         'EXPERIMENT_NAME'
assert "RUN_PREFIX = 'pokemon-20260627-v0-08d4'" in s1, 'RUN_PREFIX'
assert 'RUN_LOCAL_EVAL = False' in s1,                  'eval disabled'
s7 = ''.join(nb2['cells'][7]['source'])
assert "'option_index'" in s19,                         'option_index excluded'
assert 'turn_action_count' in s19,                      'turn_action_count still excluded'
assert 'v08d4' in s19,                                  'v08d4 comment present'
assert 'v08d4 preload' in s7,                           'preload logic in Cell[7]'
assert '_PRELOAD_CANDIDATES' in s7,                     'preload candidates in Cell[7]'
full = ''.join(''.join(c['source']) for c in nb2['cells'])
assert 'lambdarank' in full, 'lambdarank kept'
print('All sanity checks passed.')
