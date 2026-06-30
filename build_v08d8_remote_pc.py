"""Build pokemon-20260627-v0-08d8-remote-pc.ipynb from v08d2.

v08d8: finer LGBM hyperparameters — smaller LR, more rounds, more patience.

v08d1/v08d2/v08d3 all converge at iter 33-43. With num_leaves=63 and lr=0.05,
the model learns quickly but may be missing fine-grained signal.

Hypothesis: Reducing learning_rate to 0.01 (5x smaller) and increasing
num_boost_round to 2000 + early_stopping patience to 100 gives the optimizer
more room to find a better local minimum. Known as "slow LR" tuning.

Changes from v08d2:
  1. EXPERIMENT_NAME → v0_08d8_top200_lgbm_slow_lr
  2. RUN_PREFIX → pokemon-20260627-v0-08d8
  3. Cell[19]: UNKNOWN0_LGBM_PARAMS: learning_rate 0.05 → 0.01
     + num_boost_round 500 → 2000
     + early_stopping patience 50 → 100
"""
import json

SRC = '/kaggle/working/pokemon-20260627-v0-08d2.ipynb'
DST = '/kaggle/working/pokemon-20260627-v0-08d8.ipynb'

with open(SRC) as f:
    nb = json.load(f)

cells = nb['cells']

# ── Cell[1]: Config ───────────────────────────────────────────────────────────
src1 = ''.join(cells[1]['source'])

OLD_EXP = "EXPERIMENT_NAME = 'v0_08d2_top200_lgbm_noeval'"
NEW_EXP = "EXPERIMENT_NAME = 'v0_08d8_top200_lgbm_slow_lr'"
assert OLD_EXP in src1, f'EXPERIMENT_NAME not found'
src1 = src1.replace(OLD_EXP, NEW_EXP)

OLD_PFX = "RUN_PREFIX = 'pokemon-20260627-v0-08d2'"
NEW_PFX = "RUN_PREFIX = 'pokemon-20260627-v0-08d8'"
assert OLD_PFX in src1, f'RUN_PREFIX not found'
src1 = src1.replace(OLD_PFX, NEW_PFX)

cells[1]['source'] = src1.splitlines(keepends=True)

# ── Cell[19]: LGBM hyperparams — slow LR ─────────────────────────────────────
src19 = ''.join(cells[19]['source'])

# Patch learning_rate
OLD_LR = "'learning_rate': 0.05,"
NEW_LR = "'learning_rate': 0.01,   # v08d8: slow LR for finer convergence"
assert OLD_LR in src19, f'learning_rate not found: {OLD_LR!r}'
src19 = src19.replace(OLD_LR, NEW_LR)

# Patch num_boost_round
OLD_ROUNDS = "num_boost_round=500,"
NEW_ROUNDS = "num_boost_round=2000,   # v08d8: more rounds for slow LR"
assert OLD_ROUNDS in src19, f'num_boost_round not found'
src19 = src19.replace(OLD_ROUNDS, NEW_ROUNDS)

# Patch early_stopping patience (50 → 100)
OLD_ES = "_lgb_mod.early_stopping(50, verbose=False),"
NEW_ES = "_lgb_mod.early_stopping(100, verbose=False),   # v08d8: more patience"
assert OLD_ES in src19, f'early_stopping not found'
src19 = src19.replace(OLD_ES, NEW_ES)

cells[19]['source'] = src19.splitlines(keepends=True)

# ── Cell[7]: Preload episode data ─────────────────────────────────────────────
src7 = ''.join(cells[7]['source'])

OLD_PRELOAD_POINT = 'NON_EPISODE_JSON_ROWS = []\n\nif RUN_REPLAY_MINING:'
NEW_PRELOAD_POINT = (
    "NON_EPISODE_JSON_ROWS = []\n\n"
    "# v08d8: preload pre-processed episode data from prior run cache\n"
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
    "    print(f'v08d8 preload: {len(DECISION_ROWS_DF)} decisions, {len(OPTION_ROWS_DF)} options from {_pfx}')\n"
    "else:\n"
    "    print('v08d8 preload: no cache found, running episode mining from scratch')\n\n"
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

assert 'v0_08d8_top200_lgbm_slow_lr' in s1,         'EXPERIMENT_NAME'
assert "RUN_PREFIX = 'pokemon-20260627-v0-08d8'" in s1, 'RUN_PREFIX'
assert 'RUN_LOCAL_EVAL = False' in s1,               'eval disabled'
assert "'learning_rate': 0.01" in s19,               'slow LR patched'
assert 'num_boost_round=2000' in s19,                'more rounds patched'
assert 'early_stopping(100,' in s19,                 'patience=100 patched'
assert 'v08d8 preload' in s7,                        'preload logic in Cell[7]'
assert '_PRELOAD_CANDIDATES' in s7,                  'preload candidates in Cell[7]'
full = ''.join(''.join(c['source']) for c in nb2['cells'])
assert 'lambdarank' in full, 'lambdarank kept'
print('All sanity checks passed.')
