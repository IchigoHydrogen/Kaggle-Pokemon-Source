"""Build pokemon-20260627-v0-08d26-remote-pc.ipynb from v08d19.

v08d26: KO-event credit assignment.

Root cause of v08d23 failure: time-decay (gamma^steps_remaining) is a PROXY
for importance, not the real thing. Steps 0-5 of a 40-step game have near-zero
weight even if they set up a crucial KO.

v08d26 uses the ACTUAL KO EVENTS (op_prizes_left decrease) to identify which
decisions were KO-adjacent:
  - Decision causes KO within 1 step: winner_weight + 3.0 (= 7.0x)
  - Decision causes KO within 2 steps: winner_weight + 1.5 (= 5.5x)
  - Decision causes KO within 3 steps: winner_weight + 0.75 (= 4.75x)
  - Other winner decisions: winner_weight (= 4.0x, unchanged)
  - Loser decisions: 1.0x (unchanged)

No future-information leakage: _ko_credit used only as SAMPLE WEIGHT (not feature).

Changes from v08d19:
  1. EXPERIMENT_NAME → v0_08d26_kocredit
  2. RUN_PREFIX → pokemon-20260627-v0-08d26
  3. Cell[7]: Preload from v08d19 (same split)
  4. Cell[19]: Add KO credit computation to work (forward prize-change detection)
  5. Cell[19]: Modify winner weight block to use _ko_credit additive bonus
"""
import json

SRC = '/kaggle/working/pokemon-20260627-v0-08d19.ipynb'
DST = '/kaggle/working/pokemon-20260627-v0-08d26.ipynb'

with open(SRC) as f:
    nb = json.load(f)

cells = nb['cells']

# ── Cell[1]: Config ───────────────────────────────────────────────────────────
src1 = ''.join(cells[1]['source'])

OLD_EXP = "EXPERIMENT_NAME = 'v0_08d19_op_seq'"
NEW_EXP = "EXPERIMENT_NAME = 'v0_08d26_kocredit'"
assert OLD_EXP in src1
src1 = src1.replace(OLD_EXP, NEW_EXP)

OLD_PFX = "RUN_PREFIX = 'pokemon-20260627-v0-08d19'"
NEW_PFX = "RUN_PREFIX = 'pokemon-20260627-v0-08d26'"
assert OLD_PFX in src1
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
    "# v08d26: preload from v08d19 (same split)\n"
    "_PRELOAD_CANDIDATES = [\n"
    "    Path('/kaggle/working/pokemon-20260627-v0-08d19/pokemon-20260627-v0-08d19'),\n"
    "    Path('/kaggle/working/pokemon-20260627-v0-08d18/pokemon-20260627-v0-08d18'),\n"
    "]\n"
)
assert OLD_PRELOAD in src7
src7 = src7.replace(OLD_PRELOAD, NEW_PRELOAD)

OLD_PRINT = "    print(f'v08d19 preload: {len(DECISION_ROWS_DF)} decisions from {_pfx}')\n"
NEW_PRINT = "    print(f'v08d26 preload: {len(DECISION_ROWS_DF)} decisions from {_pfx}')\n"
assert OLD_PRINT in src7
src7 = src7.replace(OLD_PRINT, NEW_PRINT)

OLD_FAIL = "    print('v08d19 preload: v08d18 cache not found, mining from scratch')\n"
NEW_FAIL = "    print('v08d26 preload: no cache found, mining from scratch')\n"
assert OLD_FAIL in src7
src7 = src7.replace(OLD_FAIL, NEW_FAIL)

cells[7]['source'] = src7.splitlines(keepends=True)

# ── Cell[19]: KO credit assignment ────────────────────────────────────────────
src19 = ''.join(cells[19]['source'])

# 1. Add KO credit computation before _extra_numeric_candidates
OLD_ANCHOR = (
    "        else:\n"
    "            work['op_last_context'] = 'NONE'\n"
    "            work['steps_since_op'] = -1.0\n"
    "        # Extra features from ALAKAZAM_OPTION_MODEL_DF\n"
    "        _extra_numeric_candidates = ['supporter_played', 'stadium_played', 'energy_attached',\n"
    "                                      'my_discard_count', 'op_discard_count', 'prize_gap', 'steps_since_op']\n"
)
NEW_ANCHOR = (
    "        else:\n"
    "            work['op_last_context'] = 'NONE'\n"
    "            work['steps_since_op'] = -1.0\n"
    "        # v08d26: KO-event credit — forward-looking prize-change detection\n"
    "        # _ko_credit used only as sample weight, not as a feature (no future info leakage)\n"
    "        if 'op_prizes_left' in work.columns and 'episode_id' in work.columns:\n"
    "            _ws = work.sort_values(['episode_id', 'player_index', 'step'])\n"
    "            _g = _ws.groupby(['episode_id', 'player_index'])\n"
    "            _fwd1 = _g['op_prizes_left'].shift(-1)\n"
    "            _fwd2 = _g['op_prizes_left'].shift(-2)\n"
    "            _fwd3 = _g['op_prizes_left'].shift(-3)\n"
    "            _ws2 = _ws.copy()\n"
    "            _ws2['_ko_dist'] = 999\n"
    "            _ws2.loc[(_fwd3 < _ws2['op_prizes_left']).fillna(False), '_ko_dist'] = 3\n"
    "            _ws2.loc[(_fwd2 < _ws2['op_prizes_left']).fillna(False), '_ko_dist'] = 2\n"
    "            _ws2.loc[(_fwd1 < _ws2['op_prizes_left']).fillna(False), '_ko_dist'] = 1\n"
    "            work = work.merge(_ws2[['decision_id', '_ko_dist']], on='decision_id', how='left')\n"
    "            work['_ko_dist'] = work['_ko_dist'].fillna(999)\n"
    "            _ko_bonus_max = 3.0\n"
    "            work['_ko_credit'] = work['_ko_dist'].map(\n"
    "                lambda d: _ko_bonus_max * (0.5 ** int(d)) if int(d) < 4 else 0.0)\n"
    "            _n_ko = int((work['_ko_dist'] < 4).sum())\n"
    "            _ko_dist_str = work['_ko_dist'].replace(999, float('nan')).value_counts().sort_index().to_dict()\n"
    "            print(f'v08d26 ko_credit: {_n_ko}/{len(work)} KO-adj decisions, '\n"
    "                  f'dist={_ko_dist_str}, mean_credit={work[\"_ko_credit\"].mean():.3f}')\n"
    "        else:\n"
    "            work['_ko_credit'] = 0.0\n"
    "            print('v08d26 ko_credit: fallback (op_prizes_left missing)')\n"
    "        # Extra features from ALAKAZAM_OPTION_MODEL_DF\n"
    "        _extra_numeric_candidates = ['supporter_played', 'stadium_played', 'energy_attached',\n"
    "                                      'my_discard_count', 'op_discard_count', 'prize_gap', 'steps_since_op']\n"
)
assert OLD_ANCHOR in src19, 'anchor block not found in Cell[19]'
src19 = src19.replace(OLD_ANCHOR, NEW_ANCHOR)

# 2. Modify winner weight block to use additive _ko_credit
OLD_WT_BLOCK = (
    "                _winner_weight = 4.0\n"
    "                train_df = train_df.copy()\n"
    "                train_df['_win_wt'] = train_df['won'].fillna(0).astype(float) * (_winner_weight - 1.0) + 1.0\n"
    "                _n_winner = int((train_df['won'] == True).sum())\n"
    "                print(f'Winner-weighted training: {_n_winner}/{len(train_df)} winner rows (weight={_winner_weight}x)')\n"
)
NEW_WT_BLOCK = (
    "                _winner_weight = 4.0\n"
    "                train_df = train_df.copy()\n"
    "                # v08d26: KO-event credit — additive bonus on winner_weight\n"
    "                _winner_mask = train_df['won'].fillna(0).astype(bool)\n"
    "                _ko_cred = train_df['_ko_credit'] if '_ko_credit' in train_df.columns else 0.0\n"
    "                train_df['_win_wt'] = 1.0\n"
    "                train_df.loc[_winner_mask, '_win_wt'] = _winner_weight + _ko_cred[_winner_mask]\n"
    "                _n_winner = int(_winner_mask.sum())\n"
    "                print(f'v08d26 ko winner: {_n_winner}/{len(train_df)} rows, '\n"
    "                      f'wt=[{train_df.loc[_winner_mask,\"_win_wt\"].min():.2f},'\n"
    "                      f'{train_df.loc[_winner_mask,\"_win_wt\"].max():.2f}], '\n"
    "                      f'mean={train_df.loc[_winner_mask,\"_win_wt\"].mean():.2f}')\n"
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

assert 'v0_08d26_kocredit' in s1
assert "RUN_PREFIX = 'pokemon-20260627-v0-08d26'" in s1
assert 'v08d19' in s7
assert '_ko_credit' in s19, 'ko_credit not in Cell[19]'
assert '_ko_dist' in s19, 'ko_dist not in Cell[19]'
assert 'v08d26 ko winner' in s19, 'ko winner print not found'
assert "winner rows (weight={_winner_weight}x))" not in s19, 'old flat winner print still present'
assert 'op_last_context' in s19
assert 'prize_gap' in s19
assert '_winner_weight = 4.0' in s19
full = ''.join(''.join(c['source']) for c in nb2['cells'])
assert 'lambdarank' in full
print('All sanity checks passed.')
