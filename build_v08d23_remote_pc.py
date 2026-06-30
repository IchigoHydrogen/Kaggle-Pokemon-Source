"""Build pokemon-20260627-v0-08d23-remote-pc.ipynb from v08d19.

v08d23: Temporal credit assignment — γ^steps_remaining weighting.

PARADIGM SHIFT: First experiment that moves beyond t=0 snapshot imitation.
Instead of flat winner_weight=4x, use per-decision temporal discount weight:
  - Later decisions (close to game end) → higher weight (γ^0 ≈ 1.0)
  - Earlier decisions (far from game end) → lower weight (γ^20 ≈ 0.36)
  - Normalization preserves mean winner weight = 4x

Hypothesis: Game-deciding moments (prize races, KO sequences) happen late.
Uniform 4x dilutes signal with irrelevant early-game setup decisions.

Current state (v08d19 baseline):
  winner_weight=4x flat → top1=0.5469

Changes from v08d19:
  1. EXPERIMENT_NAME → v0_08d23_credit
  2. RUN_PREFIX → pokemon-20260627-v0-08d23
  3. Cell[7]: Preload from v08d19 (same split)
  4. Cell[19]: Replace flat winner_weight with γ^steps_remaining credit assignment
  5. Cell[1]: Keep 350k rows (same as v08d19)
"""
import json

SRC = '/kaggle/working/pokemon-20260627-v0-08d19.ipynb'
DST = '/kaggle/working/pokemon-20260627-v0-08d23.ipynb'

with open(SRC) as f:
    nb = json.load(f)

cells = nb['cells']

# ── Cell[1]: Config ───────────────────────────────────────────────────────────
src1 = ''.join(cells[1]['source'])

OLD_EXP = "EXPERIMENT_NAME = 'v0_08d19_op_seq'"
NEW_EXP = "EXPERIMENT_NAME = 'v0_08d23_credit'"
assert OLD_EXP in src1, f'EXPERIMENT_NAME not found'
src1 = src1.replace(OLD_EXP, NEW_EXP)

OLD_PFX = "RUN_PREFIX = 'pokemon-20260627-v0-08d19'"
NEW_PFX = "RUN_PREFIX = 'pokemon-20260627-v0-08d23'"
assert OLD_PFX in src1, f'RUN_PREFIX not found'
src1 = src1.replace(OLD_PFX, NEW_PFX)

cells[1]['source'] = src1.splitlines(keepends=True)

# ── Cell[7]: Preload from v08d19 ──────────────────────────────────────────────
src7 = ''.join(cells[7]['source'])

OLD_PRELOAD = (
    "# v08d19: preload from v08d18 (has correct prizes_left after bug fix)\n"
    "_PRELOAD_CANDIDATES = [\n"
    "    Path('/kaggle/working/pokemon-20260627-v0-08d18/pokemon-20260627-v0-08d18'),\n"
    "]\n"
)
NEW_PRELOAD = (
    "# v08d23: preload from v08d19 (same split)\n"
    "_PRELOAD_CANDIDATES = [\n"
    "    Path('/kaggle/working/pokemon-20260627-v0-08d19/pokemon-20260627-v0-08d19'),\n"
    "    Path('/kaggle/working/pokemon-20260627-v0-08d18/pokemon-20260627-v0-08d18'),\n"
    "]\n"
)
assert OLD_PRELOAD in src7, f'preload block not found in Cell[7]'
src7 = src7.replace(OLD_PRELOAD, NEW_PRELOAD)

OLD_PRINT = "    print(f'v08d19 preload: {len(DECISION_ROWS_DF)} decisions from {_pfx}')\n"
NEW_PRINT = "    print(f'v08d23 preload: {len(DECISION_ROWS_DF)} decisions from {_pfx}')\n"
assert OLD_PRINT in src7, f'preload success print not found'
src7 = src7.replace(OLD_PRINT, NEW_PRINT)

OLD_FAIL = "    print('v08d19 preload: v08d18 cache not found, mining from scratch')\n"
NEW_FAIL = "    print('v08d23 preload: no cache found, mining from scratch')\n"
assert OLD_FAIL in src7, f'preload fail print not found'
src7 = src7.replace(OLD_FAIL, NEW_FAIL)

cells[7]['source'] = src7.splitlines(keepends=True)

# ── Cell[19]: Temporal credit assignment (replaces flat winner_weight) ─────────
src19 = ''.join(cells[19]['source'])

OLD_WT_BLOCK = (
    "                _winner_weight = 4.0\n"
    "                train_df = train_df.copy()\n"
    "                train_df['_win_wt'] = train_df['won'].fillna(0).astype(float) * (_winner_weight - 1.0) + 1.0\n"
    "                _n_winner = int((train_df['won'] == True).sum())\n"
    "                print(f'Winner-weighted training: {_n_winner}/{len(train_df)} winner rows (weight={_winner_weight}x)')\n"
)
NEW_WT_BLOCK = (
    "                _winner_weight = 4.0  # base multiplier\n"
    "                _gamma = 0.95          # temporal discount factor\n"
    "                train_df = train_df.copy()\n"
    "                # Credit assignment: later decisions get higher weight (γ^steps_remaining)\n"
    "                _ep_max = (DECISION_ROWS_DF.groupby('episode_id')['step']\n"
    "                           .max().rename('_gmax').reset_index())\n"
    "                _sinfo = (DECISION_ROWS_DF[['decision_id', 'step', 'episode_id']]\n"
    "                          .rename(columns={'step': '_dr_step'})\n"
    "                          .merge(_ep_max, on='episode_id', how='left'))\n"
    "                train_df = train_df.merge(\n"
    "                    _sinfo[['decision_id', '_dr_step', '_gmax']],\n"
    "                    on='decision_id', how='left')\n"
    "                train_df['_srem'] = (train_df['_gmax'] - train_df['_dr_step'].fillna(0)).clip(lower=0)\n"
    "                train_df['_disc'] = _gamma ** train_df['_srem']\n"
    "                _wm = train_df['won'].fillna(0).astype(bool)\n"
    "                _md = train_df.loc[_wm, '_disc'].mean()\n"
    "                if _md > 0:\n"
    "                    train_df['_disc'] = train_df['_disc'] / _md\n"
    "                train_df['_win_wt'] = 1.0\n"
    "                train_df.loc[_wm, '_win_wt'] = train_df.loc[_wm, '_disc'] * _winner_weight\n"
    "                _n_winner = int(_wm.sum())\n"
    "                print(f'v08d23 credit: {_n_winner}/{len(train_df)} winner rows, '\n"
    "                      f'gamma={_gamma}, base={_winner_weight}x, '\n"
    "                      f'wt=[{train_df.loc[_wm, \"_win_wt\"].min():.2f},'\n"
    "                      f'{train_df.loc[_wm, \"_win_wt\"].max():.2f}], '\n"
    "                      f'mean={train_df.loc[_wm, \"_win_wt\"].mean():.2f}')\n"
)

assert OLD_WT_BLOCK in src19, f'winner_weight block not found in Cell[19]'
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

assert 'v0_08d23_credit' in s1
assert "RUN_PREFIX = 'pokemon-20260627-v0-08d23'" in s1
assert 'v08d19' in s7, 'preload from v08d19'
assert '_gamma = 0.95' in s19, 'gamma not found'
assert '_dr_step' in s19, 'credit assignment step merge not found'
assert 'steps_remaining' in s19 or '_srem' in s19, 'srem not found'
assert '_winner_weight = 4.0' in s19, 'base winner_weight not found'
assert "weight={_winner_weight}x)" not in s19, 'old flat winner_weight print still present'
assert 'op_last_context' in s19, 'op_last_context kept'
assert 'prize_gap' in s19, 'prize_gap kept'
full = ''.join(''.join(c['source']) for c in nb2['cells'])
assert 'lambdarank' in full
print('All sanity checks passed.')
