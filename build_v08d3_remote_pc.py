"""Build pokemon-20260627-v0-08d3-remote-pc.ipynb from v08d2.

v08d3: rate-weighted training — decisions from higher-rated players get more weight.

Hypothesis: Top-ranked players (rate ~1400) make systematically better decisions
than rank-200 players (rate ~988). Weighting by rate concentrates learning on
the highest-quality expert behavior.

Formula: sample_weight = (won * (wt - 1.0) + 1.0) * rate_factor
  where rate_factor = rate / mean_rate (normalized; mean_rate ≈ ~1080)
  Effectively:
    winner from rate-1400 player → weight = 4.0 * (1400/1080) ≈ 5.2x
    loser from rate-1400 player  → weight = 1.0 * (1400/1080) ≈ 1.3x
    winner from rate-988 player  → weight = 4.0 * (988/1080) ≈ 3.7x
    loser from rate-988 player   → weight = 1.0 * (988/1080) ≈ 0.9x

Changes from v08d2:
  1. EXPERIMENT_NAME → v0_08d3_top200_lgbm_rate_wt
  2. RUN_PREFIX → pokemon-20260627-v0-08d3
  3. Cell[19]: replace winner-weight formula with rate-weighted version
     OLD: train_df['_win_wt'] = train_df['won'].fillna(0).astype(float) * (_winner_weight - 1.0) + 1.0
     NEW: adds rate_factor multiplication based on 'rate' column
"""
import json

SRC = '/kaggle/working/pokemon-20260627-v0-08d2.ipynb'
DST = '/kaggle/working/pokemon-20260627-v0-08d3.ipynb'

with open(SRC) as f:
    nb = json.load(f)

cells = nb['cells']

# ── Cell[1]: Config ───────────────────────────────────────────────────────────
src1 = ''.join(cells[1]['source'])

OLD_EXP = "EXPERIMENT_NAME = 'v0_08d2_top200_lgbm_noeval'"
NEW_EXP = "EXPERIMENT_NAME = 'v0_08d3_top200_lgbm_rate_wt'"
assert OLD_EXP in src1, f'EXPERIMENT_NAME not found'
src1 = src1.replace(OLD_EXP, NEW_EXP)

OLD_PFX = "RUN_PREFIX = 'pokemon-20260627-v0-08d2'"
NEW_PFX = "RUN_PREFIX = 'pokemon-20260627-v0-08d3'"
assert OLD_PFX in src1, f'RUN_PREFIX not found'
src1 = src1.replace(OLD_PFX, NEW_PFX)

cells[1]['source'] = src1.splitlines(keepends=True)

# ── Cell[19]: Add rate-weighted training formula ──────────────────────────────
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
    "            # v08d3: rate-weighted training — higher-rated player decisions get more weight\n"
    "            if 'won' in train_df.columns:\n"
    "                _winner_weight = 4.0\n"
    "                train_df = train_df.copy()\n"
    "                _base_wt = train_df['won'].fillna(0).astype(float) * (_winner_weight - 1.0) + 1.0\n"
    "                if 'rate' in train_df.columns:\n"
    "                    _mean_rate = train_df['rate'].fillna(train_df['rate'].mean()).mean()\n"
    "                    _rate_factor = train_df['rate'].fillna(_mean_rate) / _mean_rate\n"
    "                    train_df['_win_wt'] = _base_wt * _rate_factor\n"
    "                    print(f'Rate-weighted training: mean_rate={_mean_rate:.1f}, rate_factor range=[{_rate_factor.min():.2f}, {_rate_factor.max():.2f}]')\n"
    "                else:\n"
    "                    train_df['_win_wt'] = _base_wt\n"
    "                    print('rate column not found, using winner_weight only')\n"
    "                _n_winner = int((train_df['won'] == True).sum())\n"
    "                print(f'Winner-weighted training: {_n_winner}/{len(train_df)} winner rows (base weight={_winner_weight}x, rate-scaled)')"
)
assert OLD_WT in src19, 'winner_weight formula not found in Cell[19]'
src19 = src19.replace(OLD_WT, NEW_WT)

cells[19]['source'] = src19.splitlines(keepends=True)

# ── Cell[7]: Preload episode data from v08d2 cache ────────────────────────────
# Episode processing (5600 × 4-5MB JSONs) takes 2-3h. v08d2 already processed
# the same data. Load from v08d2's parquet files to skip the expensive step.
src7 = ''.join(cells[7]['source'])

OLD_PRELOAD_POINT = 'NON_EPISODE_JSON_ROWS = []\n\nif RUN_REPLAY_MINING:'
NEW_PRELOAD_POINT = (
    "NON_EPISODE_JSON_ROWS = []\n\n"
    "# v08d3+: preload pre-processed episode data from v08d2 cache\n"
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
    "    print(f'v08d3 preload: {len(DECISION_ROWS_DF)} decisions, {len(OPTION_ROWS_DF)} options from {_pfx}')\n"
    "else:\n"
    "    print('v08d3 preload: no cache found, running episode mining from scratch')\n\n"
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

assert 'v0_08d3_top200_lgbm_rate_wt' in s1,           'EXPERIMENT_NAME'
assert "RUN_PREFIX = 'pokemon-20260627-v0-08d3'" in s1, 'RUN_PREFIX'
assert 'RUN_LOCAL_EVAL = False' in s1,                  'eval disabled'
assert 'rate_factor' in s19,                            'rate_factor in training'
assert '_mean_rate' in s19,                             '_mean_rate computed'
assert 'rate-weighted training' in s19.lower(),         'rate-weighted comment'
assert 'episodes-2026-06-26' in s1,                    '20260626 data kept'
assert 'v08d3 preload' in s7,                           'preload logic in Cell[7]'
assert '_PRELOAD_CANDIDATES' in s7,                     'preload candidates in Cell[7]'
full = ''.join(''.join(c['source']) for c in nb2['cells'])
assert 'lambdarank' in full, 'lambdarank kept'
print('All sanity checks passed.')
