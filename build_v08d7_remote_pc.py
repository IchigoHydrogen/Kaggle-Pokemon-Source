"""Build pokemon-20260627-v0-08d7-remote-pc.ipynb from v08d2.

v08d7: top50-only training (higher quality player decisions).

The top200 ranking spans rate 988-1288 (~300 points). Rate correlates with
decision quality. The bottom half of top200 might introduce noise.

Hypothesis: Using only top50 players (rank 1-50, rate ≈ 1200-1350) gives a
cleaner, higher-quality expert signal. Less data but better signal-to-noise.

Implementation: patch the is_top200 filter in Cell[19] to require
rank <= 50 (or equivalent rate threshold). The existing 'rate' column can
be used — select top half of rate distribution as proxy for top50.

Changes from v08d2:
  1. EXPERIMENT_NAME → v0_08d7_top200_lgbm_top50
  2. RUN_PREFIX → pokemon-20260627-v0-08d7
  3. Cell[19]: filter train_df to only top50 players by rate threshold
     Add: train_df = train_df[train_df['rate'] >= train_df['rate'].quantile(0.75)].copy()
     (top 25% of rate distribution ≈ rank 1-50 within top200)
"""
import json

SRC = '/kaggle/working/pokemon-20260627-v0-08d2.ipynb'
DST = '/kaggle/working/pokemon-20260627-v0-08d7.ipynb'

with open(SRC) as f:
    nb = json.load(f)

cells = nb['cells']

# ── Cell[1]: Config ───────────────────────────────────────────────────────────
src1 = ''.join(cells[1]['source'])

OLD_EXP = "EXPERIMENT_NAME = 'v0_08d2_top200_lgbm_noeval'"
NEW_EXP = "EXPERIMENT_NAME = 'v0_08d7_top200_lgbm_top50'"
assert OLD_EXP in src1, f'EXPERIMENT_NAME not found'
src1 = src1.replace(OLD_EXP, NEW_EXP)

OLD_PFX = "RUN_PREFIX = 'pokemon-20260627-v0-08d2'"
NEW_PFX = "RUN_PREFIX = 'pokemon-20260627-v0-08d7'"
assert OLD_PFX in src1, f'RUN_PREFIX not found'
src1 = src1.replace(OLD_PFX, NEW_PFX)

cells[1]['source'] = src1.splitlines(keepends=True)

# ── Cell[19]: Filter to top50 by rate threshold ───────────────────────────────
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
    "            # v08d7: top50 quality filter — keep only top 25% of rate (≈ rank 1-50 within top200)\n"
    "            if 'rate' in train_df.columns:\n"
    "                _rate_q75 = train_df['rate'].quantile(0.75)\n"
    "                _n_before = len(train_df)\n"
    "                train_df = train_df[train_df['rate'] >= _rate_q75].copy()\n"
    "                print(f'Top50 filter: {len(train_df)}/{_n_before} rows kept (rate >= {_rate_q75:.0f}, top 25%)')\n"
    "            # Winner-weighted training: winner decisions get 4x weight\n"
    "            if 'won' in train_df.columns:\n"
    "                _winner_weight = 4.0\n"
    "                train_df = train_df.copy()\n"
    "                train_df['_win_wt'] = train_df['won'].fillna(0).astype(float) * (_winner_weight - 1.0) + 1.0\n"
    "                _n_winner = int((train_df['won'] == True).sum())\n"
    "                print(f'Winner-weighted training: {_n_winner}/{len(train_df)} winner rows (weight={_winner_weight}x)')"
)
assert OLD_WT in src19, 'winner_weight formula not found in Cell[19]'
src19 = src19.replace(OLD_WT, NEW_WT)

cells[19]['source'] = src19.splitlines(keepends=True)

# ── Cell[7]: Preload episode data ─────────────────────────────────────────────
src7 = ''.join(cells[7]['source'])

OLD_PRELOAD_POINT = 'NON_EPISODE_JSON_ROWS = []\n\nif RUN_REPLAY_MINING:'
NEW_PRELOAD_POINT = (
    "NON_EPISODE_JSON_ROWS = []\n\n"
    "# v08d7: preload pre-processed episode data from prior run cache\n"
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
    "    print(f'v08d7 preload: {len(DECISION_ROWS_DF)} decisions, {len(OPTION_ROWS_DF)} options from {_pfx}')\n"
    "else:\n"
    "    print('v08d7 preload: no cache found, running episode mining from scratch')\n\n"
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

assert 'v0_08d7_top200_lgbm_top50' in s1,           'EXPERIMENT_NAME'
assert "RUN_PREFIX = 'pokemon-20260627-v0-08d7'" in s1, 'RUN_PREFIX'
assert 'RUN_LOCAL_EVAL = False' in s1,               'eval disabled'
assert 'rate_q75' in s19,                            'top50 filter present'
assert 'quantile(0.75)' in s19,                      'quantile filter'
assert 'v08d7 preload' in s7,                        'preload logic in Cell[7]'
assert '_PRELOAD_CANDIDATES' in s7,                  'preload candidates in Cell[7]'
full = ''.join(''.join(c['source']) for c in nb2['cells'])
assert 'lambdarank' in full, 'lambdarank kept'
print('All sanity checks passed.')
