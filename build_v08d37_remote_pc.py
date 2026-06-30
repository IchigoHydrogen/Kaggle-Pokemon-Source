"""Build pokemon-20260627-v0-08d37-remote-pc.ipynb from v08d34.

v08d37: option_type_winrate — option-level value signal.

CURRENT BEST: v08d34 (truncation_level=1, position_winrate) = top1=0.5485

position_winrate (v08d28): win_rate per (prize_gap, turn_bucket, archetype)
  → DECISION-level signal: same value for ALL options in a decision
  → Captures "how good is this game state overall?"

option_type_winrate (v08d37): win_rate per (prize_gap, option_type, archetype)
  → OPTION-level signal: DIFFERENT value for each option within a decision
  → Captures "when you CHOOSE this option TYPE in this game state, what's the win rate?"

KEY DIFFERENCE:
  position_winrate tells the model: "you're in a winning/losing position"
  option_type_winrate tells the model: "attacking in this position historically leads to X% wins,
                                        playing a trainer leads to Y% wins"

This is more discriminative for top1 because it differentiates BETWEEN OPTIONS,
not just between game states.

Example: prize_gap=-2, vs Mewtwo archetype:
  - ATTACK option chosen: win_rate = 0.65 (need to pressure, attacks work)
  - TRAINER option chosen: win_rate = 0.45 (setting up when behind is wrong)
  Model learns: in this state, attack > trainer

Computation:
  - Use DECISION_ROWS_DF filtered to UNKNOWN_0
  - Join option_type from work (chosen option type per decision)
  - Group by (prize_gap, option_type, opponent_archetype_norm)
  - Compute win rate from won column
  - Merge back to ALL options in work (not just chosen)
  - Each option gets its own win_rate based on its option_type

Inference proxy: option_type is directly observable, prize_gap is known,
archetype is known → NO proxy needed (all lookup keys are inference-safe).

Based on v08d34 (truncation_level=1, position_winrate, top1=0.5485 BEST).
NOT v08d36 (winner_weight=8.0 failed).

Changes from v08d34:
  1. EXPERIMENT_NAME → v0_08d37_opttype_wr
  2. RUN_PREFIX → pokemon-20260627-v0-08d37
  3. Cell[7]: Preload from v08d19 (same split)
  4. Cell[19]: Compute option_type_winrate, merge to work, add to candidates
"""
import json

SRC = '/kaggle/working/pokemon-20260627-v0-08d34.ipynb'
DST = '/kaggle/working/pokemon-20260627-v0-08d37.ipynb'

with open(SRC) as f:
    nb = json.load(f)

cells = nb['cells']

# ── Cell[1]: Config ───────────────────────────────────────────────────────────
src1 = ''.join(cells[1]['source'])

OLD_EXP = "EXPERIMENT_NAME = 'v0_08d34_trunc1'"
NEW_EXP = "EXPERIMENT_NAME = 'v0_08d37_opttype_wr'"
assert OLD_EXP in src1
src1 = src1.replace(OLD_EXP, NEW_EXP)

OLD_PFX = "RUN_PREFIX = 'pokemon-20260627-v0-08d34'"
NEW_PFX = "RUN_PREFIX = 'pokemon-20260627-v0-08d37'"
assert OLD_PFX in src1
src1 = src1.replace(OLD_PFX, NEW_PFX)

cells[1]['source'] = src1.splitlines(keepends=True)

# ── Cell[7]: Preload ──────────────────────────────────────────────────────────
src7 = ''.join(cells[7]['source'])

OLD_PRELOAD = "# v08d34: preload from v08d19 (same split)\n"
NEW_PRELOAD = "# v08d37: preload from v08d19 (same split)\n"
assert OLD_PRELOAD in src7
src7 = src7.replace(OLD_PRELOAD, NEW_PRELOAD)

OLD_PRINT = "    print(f'v08d34 preload: {len(DECISION_ROWS_DF)} decisions from {_pfx}')\n"
NEW_PRINT = "    print(f'v08d37 preload: {len(DECISION_ROWS_DF)} decisions from {_pfx}')\n"
assert OLD_PRINT in src7
src7 = src7.replace(OLD_PRINT, NEW_PRINT)

OLD_FAIL = "    print('v08d34 preload: no cache found, mining from scratch')\n"
NEW_FAIL = "    print('v08d37 preload: no cache found, mining from scratch')\n"
assert OLD_FAIL in src7
src7 = src7.replace(OLD_FAIL, NEW_FAIL)

cells[7]['source'] = src7.splitlines(keepends=True)

# ── Cell[19]: Add option_type_winrate ─────────────────────────────────────────
src19 = ''.join(cells[19]['source'])

# Insert option_type_winrate computation right before _extra_numeric_candidates
OLD_ANCHOR = (
    "        # Extra features from ALAKAZAM_OPTION_MODEL_DF\n"
    "        _extra_numeric_candidates = ['supporter_played', 'stadium_played', 'energy_attached',\n"
    "                                      'my_discard_count', 'op_discard_count', 'prize_gap', 'steps_since_op',\n"
    "                                      'position_winrate']\n"
)
NEW_ANCHOR = (
    "        # v08d37: option_type_winrate — option-level value signal\n"
    "        # Unlike position_winrate (same for all options in a decision),\n"
    "        # option_type_winrate DIFFERS per option based on option_type.\n"
    "        # Captures: 'when players CHOSE an ATTACK in prize_gap=-2 vs Mewtwo, win_rate=X'\n"
    "        _work_dec = work.drop_duplicates('decision_id')[['decision_id', 'prize_gap', 'opponent_archetype_norm']].copy()\n"
    "        _work_opted = work[work['is_chosen'] == 1][['decision_id', 'option_type']].copy()\n"
    "        _dec_ot = _work_dec.merge(_work_opted, on='decision_id', how='left')\n"
    "        # Join to DECISION_ROWS_DF to get won column\n"
    "        _dr_ot = DECISION_ROWS_DF[['decision_id', 'won']].copy()\n"
    "        _dr_ot = _dr_ot.merge(_dec_ot, on='decision_id', how='inner')\n"
    "        _ot_wr_table = (_dr_ot.groupby(['prize_gap', 'option_type', 'opponent_archetype_norm'])['won']\n"
    "                        .agg(['mean', 'count']).reset_index()\n"
    "                        .rename(columns={'mean': 'option_type_winrate', 'count': '_ot_wr_n'}))\n"
    "        _ot_wr_table = _ot_wr_table[_ot_wr_table['_ot_wr_n'] >= 5]\n"
    "        # Merge to ALL options (each option gets its own option_type_winrate by its option_type)\n"
    "        work = work.merge(\n"
    "            _ot_wr_table[['prize_gap', 'option_type', 'opponent_archetype_norm', 'option_type_winrate']],\n"
    "            on=['prize_gap', 'option_type', 'opponent_archetype_norm'], how='left')\n"
    "        work['option_type_winrate'] = work['option_type_winrate'].fillna(0.5)\n"
    "        _ot_wr_cells = len(_ot_wr_table)\n"
    "        _ot_std = work['option_type_winrate'].std()\n"
    "        print(f'v08d37 option_type_winrate: std={_ot_std:.4f}, cells={_ot_wr_cells}, '\n"
    "              f'work_rows={len(work)} (should ~350k)')\n"
    "        # Extra features from ALAKAZAM_OPTION_MODEL_DF\n"
    "        _extra_numeric_candidates = ['supporter_played', 'stadium_played', 'energy_attached',\n"
    "                                      'my_discard_count', 'op_discard_count', 'prize_gap', 'steps_since_op',\n"
    "                                      'position_winrate', 'option_type_winrate']\n"
)
assert OLD_ANCHOR in src19, 'anchor block not found in Cell[19]'
src19 = src19.replace(OLD_ANCHOR, NEW_ANCHOR)

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

assert 'v0_08d37_opttype_wr' in s1
assert "RUN_PREFIX = 'pokemon-20260627-v0-08d37'" in s1
assert 'v08d19' in s7
assert 'option_type_winrate' in s19
assert "'option_type_winrate'" in s19
assert 'option_type_winrate: std=' in s19
assert 'position_winrate' in s19, 'position_winrate from v08d34 must be kept'
assert 'lambdarank_truncation_level' in s19
assert "lambdarank_truncation_level': 1" in s19
assert '_winner_weight = 4.0' in s19
assert 'op_last_context' in s19
full = ''.join(''.join(c['source']) for c in nb2['cells'])
assert 'lambdarank' in full
print('All sanity checks passed.')
