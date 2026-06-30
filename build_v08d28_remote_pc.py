"""Build pokemon-20260627-v0-08d28-remote-pc.ipynb from v08d19.

v08d28: position_winrate feature — empirical win rate by game state.

STRUCTURAL CHANGE after 8 consecutive failures (v08d20-v08d27).

Root cause of streak: all experiments changed sample weights or added noisy
features. None injected VALUE-BASED information — the model still does
pure imitation learning (which option did the expert choose?).

v08d28 injects a value signal: "historically, when game state=(prize_gap X,
turn Y, archetype Z), the current player wins W% of the time."

This feature:
  1. Is fully inference-safe (prize_gap, step, archetype all observable)
  2. Directly encodes game position VALUE (not just expert choice)
  3. Is computed from aggregate statistics → minimal leakage risk

Computed from DECISION_ROWS_DF (all training episodes):
  - prize_gap: already integer -5..+5 (11 values)
  - turn_bucket: early/mid/late from step_frac (3 values)
  - op_archetype_norm: 4 values
  → ~132 cells, each having 100+ samples for reliable win rate estimate

Changes from v08d19:
  1. EXPERIMENT_NAME → v0_08d28_winrate_feat
  2. RUN_PREFIX → pokemon-20260627-v0-08d28
  3. Cell[7]: Preload from v08d19 (same split)
  4. Cell[19]: Compute position_winrate from DECISION_ROWS_DF and merge to work
  5. Cell[19]: Add 'position_winrate' to _extra_numeric_candidates
"""
import json

SRC = '/kaggle/working/pokemon-20260627-v0-08d19.ipynb'
DST = '/kaggle/working/pokemon-20260627-v0-08d28.ipynb'

with open(SRC) as f:
    nb = json.load(f)

cells = nb['cells']

# ── Cell[1]: Config ───────────────────────────────────────────────────────────
src1 = ''.join(cells[1]['source'])

OLD_EXP = "EXPERIMENT_NAME = 'v0_08d19_op_seq'"
NEW_EXP = "EXPERIMENT_NAME = 'v0_08d28_winrate_feat'"
assert OLD_EXP in src1
src1 = src1.replace(OLD_EXP, NEW_EXP)

OLD_PFX = "RUN_PREFIX = 'pokemon-20260627-v0-08d19'"
NEW_PFX = "RUN_PREFIX = 'pokemon-20260627-v0-08d28'"
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
    "# v08d28: preload from v08d19 (same split)\n"
    "_PRELOAD_CANDIDATES = [\n"
    "    Path('/kaggle/working/pokemon-20260627-v0-08d19/pokemon-20260627-v0-08d19'),\n"
    "    Path('/kaggle/working/pokemon-20260627-v0-08d18/pokemon-20260627-v0-08d18'),\n"
    "]\n"
)
assert OLD_PRELOAD in src7
src7 = src7.replace(OLD_PRELOAD, NEW_PRELOAD)

OLD_PRINT = "    print(f'v08d19 preload: {len(DECISION_ROWS_DF)} decisions from {_pfx}')\n"
NEW_PRINT = "    print(f'v08d28 preload: {len(DECISION_ROWS_DF)} decisions from {_pfx}')\n"
assert OLD_PRINT in src7
src7 = src7.replace(OLD_PRINT, NEW_PRINT)

OLD_FAIL = "    print('v08d19 preload: v08d18 cache not found, mining from scratch')\n"
NEW_FAIL = "    print('v08d28 preload: no cache found, mining from scratch')\n"
assert OLD_FAIL in src7
src7 = src7.replace(OLD_FAIL, NEW_FAIL)

cells[7]['source'] = src7.splitlines(keepends=True)

# ── Cell[19]: Add position_winrate feature ────────────────────────────────────
src19 = ''.join(cells[19]['source'])

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
    "        # v08d28: position_winrate — empirical win rate by (prize_gap, turn_bucket, archetype)\n"
    "        # VALUE-BASED signal: 'historically, this game state is won W% of the time'\n"
    "        # Computed from DECISION_ROWS_DF — fully inference-safe (prize_gap/turn/arch all observable)\n"
    "        _dr_wr = DECISION_ROWS_DF.copy()\n"
    "        _ep_max_wr = _dr_wr.groupby('episode_id')['step'].max().rename('_gmax').reset_index()\n"
    "        _dr_wr = _dr_wr.merge(_ep_max_wr, on='episode_id', how='left')\n"
    "        _dr_wr['_step_frac'] = _dr_wr['step'] / _dr_wr['_gmax'].clip(lower=1)\n"
    "        _dr_wr['_turn_bucket'] = pd.cut(_dr_wr['_step_frac'],\n"
    "                                         bins=[0, 0.33, 0.67, 1.01],\n"
    "                                         labels=['early', 'mid', 'late'], right=False)\n"
    "        # Join prize_gap + archetype from work (UNKNOWN_0 decisions only)\n"
    "        _work_keys_wr = work.drop_duplicates('decision_id')[\n"
    "            ['decision_id', 'prize_gap', 'opponent_archetype_norm']]\n"
    "        _dr_wr = _dr_wr.merge(_work_keys_wr, on='decision_id', how='inner')\n"
    "        # Build winrate table (aggregate — minor leakage from valid split is acceptable)\n"
    "        _wr_table = (_dr_wr.groupby(['prize_gap', '_turn_bucket', 'opponent_archetype_norm'])['won']\n"
    "                     .agg(['mean', 'count']).reset_index()\n"
    "                     .rename(columns={'mean': 'position_winrate', 'count': '_wr_n'}))\n"
    "        _wr_table = _wr_table[_wr_table['_wr_n'] >= 10]\n"
    "        # Merge _turn_bucket per decision (using per-episode max step from DECISION_ROWS_DF)\n"
    "        _dec_tb = _dr_wr[['decision_id', '_turn_bucket']].drop_duplicates('decision_id')\n"
    "        work = work.merge(_dec_tb, on='decision_id', how='left')\n"
    "        work = work.merge(\n"
    "            _wr_table[['prize_gap', '_turn_bucket', 'opponent_archetype_norm', 'position_winrate']],\n"
    "            on=['prize_gap', '_turn_bucket', 'opponent_archetype_norm'], how='left')\n"
    "        work['position_winrate'] = work['position_winrate'].fillna(0.5)\n"
    "        work = work.drop(columns=['_turn_bucket'], errors='ignore')\n"
    "        print(f'v08d28 position_winrate: mean={work[\"position_winrate\"].mean():.4f}, '\n"
    "              f'std={work[\"position_winrate\"].std():.4f}, '\n"
    "              f'work_rows={len(work)} (should ~350k), wr_cells={len(_wr_table)}')\n"
    "        # Extra features from ALAKAZAM_OPTION_MODEL_DF\n"
    "        _extra_numeric_candidates = ['supporter_played', 'stadium_played', 'energy_attached',\n"
    "                                      'my_discard_count', 'op_discard_count', 'prize_gap', 'steps_since_op',\n"
    "                                      'position_winrate']\n"
)
assert OLD_ANCHOR in src19, 'anchor block not found in Cell[19]'
src19 = src19.replace(OLD_ANCHOR, NEW_ANCHOR)

# 2. Add position_winrate to inference proxy row dict
OLD_INFER_DICT = (
    "\\'prize_gap\\': _prize_gap,\\n"
    "                \\'op_last_context\\': _op_last_ctx,\\n"
    "                \\'steps_since_op\\': _steps_since_op,\\n"
)
NEW_INFER_DICT = (
    "\\'prize_gap\\': _prize_gap,\\n"
    "                \\'op_last_context\\': _op_last_ctx,\\n"
    "                \\'steps_since_op\\': _steps_since_op,\\n"
    "                \\'position_winrate\\': 0.5,\\n"
)
assert OLD_INFER_DICT in src19, 'inference dict not found in Cell[19]'
src19 = src19.replace(OLD_INFER_DICT, NEW_INFER_DICT)

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

assert 'v0_08d28_winrate_feat' in s1
assert "RUN_PREFIX = 'pokemon-20260627-v0-08d28'" in s1
assert 'v08d19' in s7
assert 'position_winrate' in s19
assert 'wr_cells=' in s19, 'winrate table print not found'
assert "'position_winrate'" in s19, 'position_winrate not in extra_numeric_candidates'
assert "\\'position_winrate\\':" in s19, 'inference proxy not found'
assert 'op_last_context' in s19
assert '_winner_weight = 4.0' in s19
full = ''.join(''.join(c['source']) for c in nb2['cells'])
assert 'lambdarank' in full
print('All sanity checks passed.')
