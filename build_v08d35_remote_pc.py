"""Build pokemon-20260627-v0-08d35-remote-pc.ipynb from v08d34.

v08d35: turn_action_count + lambdarank_truncation_level=1.

v08d33 showed: turn_action_count with truncation_level=30 (default)
  → top1=-0.0007, AUC=+0.0101 (helps ranking, hurts top1)

v08d34 showed: truncation_level=1 alone
  → top1=+0.0008 NEW BEST (AUC down as expected)

QUESTION: Does turn_action_count ALSO hurt top1 under truncation_level=1,
or does the focused gradient change its contribution?

With truncation_level=1, the model is learning "what features predict
the expert's #1 choice" exclusively. turn_action_count captures intra-turn
phase — if early/late turn action systematically predicts different #1 choices,
it could now HELP top1 specifically.

With truncation_level=30: the gradient signal was diffuse — turn_action_count
improved mid-range ranking (boosting AUC) but added noise to top-1 gradient.
With truncation_level=1: the gradient is pure top-1 signal — turn_action_count
may now cleanly help or be irrelevant.

Based on v08d34 (lambdarank_truncation_level=1, position_winrate, NEW BEST).

Changes from v08d34:
  1. EXPERIMENT_NAME → v0_08d35_tac_trunc1
  2. RUN_PREFIX → pokemon-20260627-v0-08d35
  3. Cell[7]: Preload from v08d19 (same split)
  4. Cell[19]: Add turn_action_count to _extra_numeric_candidates
"""
import json

SRC = '/kaggle/working/pokemon-20260627-v0-08d34.ipynb'
DST = '/kaggle/working/pokemon-20260627-v0-08d35.ipynb'

with open(SRC) as f:
    nb = json.load(f)

cells = nb['cells']

# ── Cell[1]: Config ───────────────────────────────────────────────────────────
src1 = ''.join(cells[1]['source'])

OLD_EXP = "EXPERIMENT_NAME = 'v0_08d34_trunc1'"
NEW_EXP = "EXPERIMENT_NAME = 'v0_08d35_tac_trunc1'"
assert OLD_EXP in src1
src1 = src1.replace(OLD_EXP, NEW_EXP)

OLD_PFX = "RUN_PREFIX = 'pokemon-20260627-v0-08d34'"
NEW_PFX = "RUN_PREFIX = 'pokemon-20260627-v0-08d35'"
assert OLD_PFX in src1
src1 = src1.replace(OLD_PFX, NEW_PFX)

cells[1]['source'] = src1.splitlines(keepends=True)

# ── Cell[7]: Preload ──────────────────────────────────────────────────────────
src7 = ''.join(cells[7]['source'])

OLD_PRELOAD = "# v08d34: preload from v08d19 (same split)\n"
NEW_PRELOAD = "# v08d35: preload from v08d19 (same split)\n"
assert OLD_PRELOAD in src7
src7 = src7.replace(OLD_PRELOAD, NEW_PRELOAD)

OLD_PRINT = "    print(f'v08d34 preload: {len(DECISION_ROWS_DF)} decisions from {_pfx}')\n"
NEW_PRINT = "    print(f'v08d35 preload: {len(DECISION_ROWS_DF)} decisions from {_pfx}')\n"
assert OLD_PRINT in src7
src7 = src7.replace(OLD_PRINT, NEW_PRINT)

OLD_FAIL = "    print('v08d34 preload: no cache found, mining from scratch')\n"
NEW_FAIL = "    print('v08d35 preload: no cache found, mining from scratch')\n"
assert OLD_FAIL in src7
src7 = src7.replace(OLD_FAIL, NEW_FAIL)

cells[7]['source'] = src7.splitlines(keepends=True)

# ── Cell[19]: Add turn_action_count to _extra_numeric_candidates ─────────────
src19 = ''.join(cells[19]['source'])

OLD_CANDS = (
    "        _extra_numeric_candidates = ['supporter_played', 'stadium_played', 'energy_attached',\n"
    "                                      'my_discard_count', 'op_discard_count', 'prize_gap', 'steps_since_op',\n"
    "                                      'position_winrate']\n"
)
NEW_CANDS = (
    "        # v08d35: add turn_action_count (test under truncation_level=1 environment)\n"
    "        _extra_numeric_candidates = ['supporter_played', 'stadium_played', 'energy_attached',\n"
    "                                      'my_discard_count', 'op_discard_count', 'prize_gap', 'steps_since_op',\n"
    "                                      'position_winrate', 'turn_action_count']\n"
)
assert OLD_CANDS in src19, '_extra_numeric_candidates not found in Cell[19]'
src19 = src19.replace(OLD_CANDS, NEW_CANDS)

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

assert 'v0_08d35_tac_trunc1' in s1
assert "RUN_PREFIX = 'pokemon-20260627-v0-08d35'" in s1
assert 'v08d19' in s7
assert 'turn_action_count' in s19
assert 'lambdarank_truncation_level' in s19
assert "lambdarank_truncation_level': 1" in s19
assert 'position_winrate' in s19
assert 'op_last_context' in s19
assert '_winner_weight = 4.0' in s19
full = ''.join(''.join(c['source']) for c in nb2['cells'])
assert 'lambdarank' in full
print('All sanity checks passed.')
