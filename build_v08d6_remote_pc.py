"""Build pokemon-20260627-v0-08d6-remote-pc.ipynb from v08d2.

v08d6: winner_weight=8x (stronger emphasis on winning decisions).

Hypothesis: v08d1 used winner_weight=4x (from v05d25). If the optimal weight
is higher, the LambdaRank signal from winning decisions will dominate more.
Doubling to 8x tests whether stronger winner emphasis helps.

Bracket: v08d5 = 1x (winner-only), v08d2/v08d1 = 4x, v08d6 = 8x.
Together these three bracket the winner-weight search space.

Changes from v08d2:
  1. EXPERIMENT_NAME → v0_08d6_top200_lgbm_winner_wt8
  2. RUN_PREFIX → pokemon-20260627-v0-08d6
  3. Cell[19]: change _winner_weight from 4.0 to 8.0
"""
import json

SRC = '/kaggle/working/pokemon-20260627-v0-08d2.ipynb'
DST = '/kaggle/working/pokemon-20260627-v0-08d6.ipynb'

with open(SRC) as f:
    nb = json.load(f)

cells = nb['cells']

# ── Cell[1]: Config ───────────────────────────────────────────────────────────
src1 = ''.join(cells[1]['source'])

OLD_EXP = "EXPERIMENT_NAME = 'v0_08d2_top200_lgbm_noeval'"
NEW_EXP = "EXPERIMENT_NAME = 'v0_08d6_top200_lgbm_winner_wt8'"
assert OLD_EXP in src1, f'EXPERIMENT_NAME not found'
src1 = src1.replace(OLD_EXP, NEW_EXP)

OLD_PFX = "RUN_PREFIX = 'pokemon-20260627-v0-08d2'"
NEW_PFX = "RUN_PREFIX = 'pokemon-20260627-v0-08d6'"
assert OLD_PFX in src1, f'RUN_PREFIX not found'
src1 = src1.replace(OLD_PFX, NEW_PFX)

cells[1]['source'] = src1.splitlines(keepends=True)

# ── Cell[19]: Change winner_weight from 4.0 to 8.0 ──────────────────────────
src19 = ''.join(cells[19]['source'])

OLD_WT = (
    "            # Winner-weighted training: winner decisions get 2x weight\n"
    "            if 'won' in train_df.columns:\n"
    "                _winner_weight = 4.0\n"
    "                train_df = train_df.copy()\n"
    "                train_df['_win_wt'] = train_df['won'].fillna(0).astype(float) * (_winner_weight - 1.0) + 1.0\n"
    "                _n_winner = int((train_df['won'] == True).sum())\n"
    "                print(f'Winner-weighted training: {_n_winner}/{len(train_df)} winner rows (weight={_winner_weight}x)')"
)
NEW_WT = (
    "            # v08d6: winner_weight=8x — double the 4x used in v08d1/v08d2\n"
    "            if 'won' in train_df.columns:\n"
    "                _winner_weight = 8.0\n"
    "                train_df = train_df.copy()\n"
    "                train_df['_win_wt'] = train_df['won'].fillna(0).astype(float) * (_winner_weight - 1.0) + 1.0\n"
    "                _n_winner = int((train_df['won'] == True).sum())\n"
    "                print(f'Winner-weighted training: {_n_winner}/{len(train_df)} winner rows (weight={_winner_weight}x)')"
)
assert OLD_WT in src19, 'winner_weight formula not found in Cell[19]'
src19 = src19.replace(OLD_WT, NEW_WT)

cells[19]['source'] = src19.splitlines(keepends=True)

# ── Cell[7]: Preload episode data (v08d2 → v08d1 fallback) ───────────────────
src7 = ''.join(cells[7]['source'])

OLD_PRELOAD_POINT = 'NON_EPISODE_JSON_ROWS = []\n\nif RUN_REPLAY_MINING:'
NEW_PRELOAD_POINT = (
    "NON_EPISODE_JSON_ROWS = []\n\n"
    "# v08d6: preload pre-processed episode data from prior run cache\n"
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
    "    print(f'v08d6 preload: {len(DECISION_ROWS_DF)} decisions, {len(OPTION_ROWS_DF)} options from {_pfx}')\n"
    "else:\n"
    "    print('v08d6 preload: no cache found, running episode mining from scratch')\n\n"
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

assert 'v0_08d6_top200_lgbm_winner_wt8' in s1,      'EXPERIMENT_NAME'
assert "RUN_PREFIX = 'pokemon-20260627-v0-08d6'" in s1, 'RUN_PREFIX'
assert 'RUN_LOCAL_EVAL = False' in s1,               'eval disabled'
assert '_winner_weight = 8.0' in s19,               'winner_weight=8.0'
assert 'winner_weight=8x' in s19.lower() or 'v08d6' in s19, 'v08d6 comment'
assert 'v08d6 preload' in s7,                        'preload logic in Cell[7]'
assert '_PRELOAD_CANDIDATES' in s7,                  'preload candidates in Cell[7]'
full = ''.join(''.join(c['source']) for c in nb2['cells'])
assert 'lambdarank' in full, 'lambdarank kept'
print('All sanity checks passed.')
