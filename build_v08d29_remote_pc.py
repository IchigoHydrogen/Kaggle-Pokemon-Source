"""Build pokemon-20260627-v0-08d29-remote-pc.ipynb from v08d28.

v08d29: Winners-only training (based on v08d28 which has position_winrate).

HYPOTHESIS: The current 4x winner_weight still includes loser decisions (46%
of training rows). Even with lower weight, loser choices may act as noise —
training the model that "in this game state, options A/B/C were chosen
(by a losing player)" could confuse the ranking signal.

APPROACH: Filter train_df to winner rows ONLY before LambdaRank training.
  - Keep: all options from WINNING player's decisions (won==True)
  - Drop: all options from LOSING player's decisions (won==False or won==None)
  - Result: ~54% of current training rows (161k → 161k winner rows out of 298k total)
  - LambdaRank still works: within each decision group, is_chosen=1 (winner's choice)
    vs is_chosen=0 (winner's unchosen options). Ranking signal is clean.

This is structurally different from all prior experiments — changes WHAT we train
on, not how we weight it.

Based on v08d28 (includes position_winrate feature, +0.0008 improvement).

Changes from v08d28:
  1. EXPERIMENT_NAME → v0_08d29_winners_only
  2. RUN_PREFIX → pokemon-20260627-v0-08d29
  3. Cell[7]: Preload from v08d19 (same split)
  4. Cell[19]: Replace winner weight block with winners-only filter
"""
import json

SRC = '/kaggle/working/pokemon-20260627-v0-08d28.ipynb'
DST = '/kaggle/working/pokemon-20260627-v0-08d29.ipynb'

with open(SRC) as f:
    nb = json.load(f)

cells = nb['cells']

# ── Cell[1]: Config ───────────────────────────────────────────────────────────
src1 = ''.join(cells[1]['source'])

OLD_EXP = "EXPERIMENT_NAME = 'v0_08d28_winrate_feat'"
NEW_EXP = "EXPERIMENT_NAME = 'v0_08d29_winners_only'"
assert OLD_EXP in src1
src1 = src1.replace(OLD_EXP, NEW_EXP)

OLD_PFX = "RUN_PREFIX = 'pokemon-20260627-v0-08d28'"
NEW_PFX = "RUN_PREFIX = 'pokemon-20260627-v0-08d29'"
assert OLD_PFX in src1
src1 = src1.replace(OLD_PFX, NEW_PFX)

cells[1]['source'] = src1.splitlines(keepends=True)

# ── Cell[7]: Preload from v08d19 ──────────────────────────────────────────────
src7 = ''.join(cells[7]['source'])

OLD_PRELOAD = (
    "# v08d28: preload from v08d19 (same split)\n"
    "_PRELOAD_CANDIDATES = [\n"
    "    Path('/kaggle/working/pokemon-20260627-v0-08d19/pokemon-20260627-v0-08d19'),\n"
    "    Path('/kaggle/working/pokemon-20260627-v0-08d18/pokemon-20260627-v0-08d18'),\n"
    "]\n"
)
NEW_PRELOAD = (
    "# v08d29: preload from v08d19 (same split)\n"
    "_PRELOAD_CANDIDATES = [\n"
    "    Path('/kaggle/working/pokemon-20260627-v0-08d19/pokemon-20260627-v0-08d19'),\n"
    "    Path('/kaggle/working/pokemon-20260627-v0-08d18/pokemon-20260627-v0-08d18'),\n"
    "]\n"
)
assert OLD_PRELOAD in src7
src7 = src7.replace(OLD_PRELOAD, NEW_PRELOAD)

OLD_PRINT = "    print(f'v08d28 preload: {len(DECISION_ROWS_DF)} decisions from {_pfx}')\n"
NEW_PRINT = "    print(f'v08d29 preload: {len(DECISION_ROWS_DF)} decisions from {_pfx}')\n"
assert OLD_PRINT in src7
src7 = src7.replace(OLD_PRINT, NEW_PRINT)

OLD_FAIL = "    print('v08d28 preload: no cache found, mining from scratch')\n"
NEW_FAIL = "    print('v08d29 preload: no cache found, mining from scratch')\n"
assert OLD_FAIL in src7
src7 = src7.replace(OLD_FAIL, NEW_FAIL)

cells[7]['source'] = src7.splitlines(keepends=True)

# ── Cell[19]: Replace winner weight block with winners-only filter ─────────────
src19 = ''.join(cells[19]['source'])

OLD_WT_BLOCK = (
    "                _winner_weight = 4.0\n"
    "                train_df = train_df.copy()\n"
    "                train_df['_win_wt'] = train_df['won'].fillna(0).astype(float) * (_winner_weight - 1.0) + 1.0\n"
    "                _n_winner = int((train_df['won'] == True).sum())\n"
    "                print(f'Winner-weighted training: {_n_winner}/{len(train_df)} winner rows (weight={_winner_weight}x)')\n"
)
NEW_WT_BLOCK = (
    "                # v08d29: winners-only training — drop all loser decisions\n"
    "                train_df = train_df.copy()\n"
    "                _winner_mask = train_df['won'].fillna(False).astype(bool)\n"
    "                _n_loser = int((~_winner_mask).sum())\n"
    "                train_df = train_df[_winner_mask].copy()\n"
    "                train_df['_win_wt'] = 1.0\n"
    "                _n_winner = int(_winner_mask.sum())\n"
    "                print(f'v08d29 winners-only: {_n_winner} rows kept, '\n"
    "                      f'{_n_loser} loser rows dropped ({100*_n_loser/(_n_winner+_n_loser):.1f}%)')\n"
)
assert OLD_WT_BLOCK in src19, 'winner_weight block not found in Cell[19]'
src19 = src19.replace(OLD_WT_BLOCK, NEW_WT_BLOCK)

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

assert 'v0_08d29_winners_only' in s1
assert "RUN_PREFIX = 'pokemon-20260627-v0-08d29'" in s1
assert 'v08d19' in s7
assert 'winners-only' in s19
assert 'v08d29 winners-only' in s19
assert 'position_winrate' in s19, 'position_winrate from v08d28 must be kept'
assert "winner rows (weight={_winner_weight}x))" not in s19, 'old winner print still there'
assert 'op_last_context' in s19
full = ''.join(''.join(c['source']) for c in nb2['cells'])
assert 'lambdarank' in full
print('All sanity checks passed.')
