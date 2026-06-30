"""Build pokemon-20260627-v0-08d43-remote-pc.ipynb from v08d34.

v08d43: position_winrate with num_options_bucket — finer-grained value signal.

CONTEXT: We've exhausted major optimization axes. v08d34 (lambdarank+trunc1,
position_winrate) = 0.5485 is the local best. This is likely the final
experiment before re-evaluating direction.

position_winrate (v08d28): groups by (prize_gap, turn_bucket, archetype) = 84 cells
  HYPOTHESIS: Same game state (same prize_gap/turn/archetype) but DIFFERENT number
  of options available might have different win rates.
  - Few options (2-4): narrow decision, expert choice is clearer → higher win rate?
  - Many options (10+): complex decision, many good choices → lower win rate?
  - num_options is DECISION-level (same for all options in a decision)
  - Adding it to the groupby is SAFE (no target leakage)

num_options_bucket: bin num_options into 3 groups
  - small: 2-5 options
  - medium: 6-10 options
  - large: 11+ options

Extended cells: (11 prize_gap) × (3 turn_bucket) × (4 archetype) × (3 num_bucket) = 396
  With 30k decisions, avg ~75 per cell — enough for reliable estimates.
  Currently position_winrate has 84 filled cells with ≥10 samples.
  Extended version might have ~250 filled cells (3x more granular).

Expected effect: the model gets a more specific "value of this game state"
signal that also captures decision complexity.

Based on v08d34 (truncation_level=1, position_winrate, top1=0.5485 BEST).

Changes from v08d34:
  1. EXPERIMENT_NAME → v0_08d43_wr_nopts
  2. RUN_PREFIX → pokemon-20260627-v0-08d43
  3. Cell[7]: Preload from v08d19 (same split)
  4. Cell[19]: Add num_options_bucket to position_winrate groupby
"""
import json

SRC = '/kaggle/working/pokemon-20260627-v0-08d34.ipynb'
DST = '/kaggle/working/pokemon-20260627-v0-08d43.ipynb'

with open(SRC) as f:
    nb = json.load(f)

cells = nb['cells']

# ── Cell[1]: Config ───────────────────────────────────────────────────────────
src1 = ''.join(cells[1]['source'])

OLD_EXP = "EXPERIMENT_NAME = 'v0_08d34_trunc1'"
NEW_EXP = "EXPERIMENT_NAME = 'v0_08d43_wr_nopts'"
assert OLD_EXP in src1
src1 = src1.replace(OLD_EXP, NEW_EXP)

OLD_PFX = "RUN_PREFIX = 'pokemon-20260627-v0-08d34'"
NEW_PFX = "RUN_PREFIX = 'pokemon-20260627-v0-08d43'"
assert OLD_PFX in src1
src1 = src1.replace(OLD_PFX, NEW_PFX)

cells[1]['source'] = src1.splitlines(keepends=True)

# ── Cell[7]: Preload ──────────────────────────────────────────────────────────
src7 = ''.join(cells[7]['source'])

OLD_PRELOAD = "# v08d34: preload from v08d19 (same split)\n"
NEW_PRELOAD = "# v08d43: preload from v08d19 (same split)\n"
assert OLD_PRELOAD in src7
src7 = src7.replace(OLD_PRELOAD, NEW_PRELOAD)

OLD_PRINT = "    print(f'v08d34 preload: {len(DECISION_ROWS_DF)} decisions from {_pfx}')\n"
NEW_PRINT = "    print(f'v08d43 preload: {len(DECISION_ROWS_DF)} decisions from {_pfx}')\n"
assert OLD_PRINT in src7
src7 = src7.replace(OLD_PRINT, NEW_PRINT)

OLD_FAIL = "    print('v08d34 preload: no cache found, mining from scratch')\n"
NEW_FAIL = "    print('v08d43 preload: no cache found, mining from scratch')\n"
assert OLD_FAIL in src7
src7 = src7.replace(OLD_FAIL, NEW_FAIL)

cells[7]['source'] = src7.splitlines(keepends=True)

# ── Cell[19]: Extend position_winrate with num_options_bucket ─────────────────
src19 = ''.join(cells[19]['source'])

# Find the position_winrate block and extend it
OLD_WR_BLOCK = (
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
)
NEW_WR_BLOCK = (
    "        _dr_wr['_turn_bucket'] = pd.cut(_dr_wr['_step_frac'],\n"
    "                                         bins=[0, 0.33, 0.67, 1.01],\n"
    "                                         labels=['early', 'mid', 'late'], right=False)\n"
    "        # Join prize_gap + archetype + num_options from work (UNKNOWN_0 decisions only)\n"
    "        # v08d43: add num_options_bucket for finer-grained win rate\n"
    "        _work_keys_wr = work.drop_duplicates('decision_id')[\n"
    "            ['decision_id', 'prize_gap', 'opponent_archetype_norm', 'num_options']].copy()\n"
    "        _work_keys_wr['_num_opts_bucket'] = pd.cut(_work_keys_wr['num_options'],\n"
    "                                                    bins=[0, 5, 10, 100],\n"
    "                                                    labels=['small', 'medium', 'large'],\n"
    "                                                    right=True)\n"
    "        _dr_wr = _dr_wr.merge(_work_keys_wr, on='decision_id', how='inner')\n"
    "        # Build winrate table with num_options_bucket as 4th dimension\n"
    "        _wr_table = (_dr_wr.groupby(['prize_gap', '_turn_bucket', 'opponent_archetype_norm',\n"
    "                                     '_num_opts_bucket'])['won']\n"
    "                     .agg(['mean', 'count']).reset_index()\n"
    "                     .rename(columns={'mean': 'position_winrate', 'count': '_wr_n'}))\n"
    "        _wr_table = _wr_table[_wr_table['_wr_n'] >= 5]  # lower threshold for more cells\n"
    "        # Merge _turn_bucket + _num_opts_bucket per decision\n"
    "        _dec_tb = _dr_wr[['decision_id', '_turn_bucket', '_num_opts_bucket']].drop_duplicates('decision_id')\n"
    "        work = work.merge(_dec_tb, on='decision_id', how='left')\n"
    "        work = work.merge(\n"
    "            _wr_table[['prize_gap', '_turn_bucket', 'opponent_archetype_norm',\n"
    "                       '_num_opts_bucket', 'position_winrate']],\n"
    "            on=['prize_gap', '_turn_bucket', 'opponent_archetype_norm', '_num_opts_bucket'],\n"
    "            how='left')\n"
    "        work['position_winrate'] = work['position_winrate'].fillna(0.5)\n"
    "        work = work.drop(columns=['_turn_bucket', '_num_opts_bucket'], errors='ignore')\n"
    "        print(f'v08d43 position_winrate+nopts: mean={work[\"position_winrate\"].mean():.4f}, '\n"
    "              f'std={work[\"position_winrate\"].std():.4f}, '\n"
    "              f'work_rows={len(work)} (should ~350k), wr_cells={len(_wr_table)}')\n"
)
assert OLD_WR_BLOCK in src19, 'position_winrate block not found in Cell[19]'
src19 = src19.replace(OLD_WR_BLOCK, NEW_WR_BLOCK)

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

assert 'v0_08d43_wr_nopts' in s1
assert "RUN_PREFIX = 'pokemon-20260627-v0-08d43'" in s1
assert 'v08d19' in s7
assert '_num_opts_bucket' in s19
assert 'position_winrate+nopts' in s19
assert 'position_winrate' in s19
assert 'lambdarank_truncation_level' in s19
assert "lambdarank_truncation_level': 1" in s19
assert '_winner_weight = 4.0' in s19
assert 'op_last_context' in s19
full = ''.join(''.join(c['source']) for c in nb2['cells'])
assert 'lambdarank' in full
print('All sanity checks passed.')
