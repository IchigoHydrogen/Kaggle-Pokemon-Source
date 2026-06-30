"""Build pokemon-20260627-v0-08d39-remote-pc.ipynb from v08d34.

v08d39: op_hand_count + my_hand_count on trunc1 base.

CONFIRMED: truncation_level=1 is optimal (0.5485, best so far).
turn_action_count consistently hurts top1 (even with trunc1).
Features that hurt top1 are: option-level features correlated with is_chosen.

Features that are SAFE for trunc1 (decision-level, NOT derived from is_chosen):
  - op_hand_count: opponent's hand size (nunique=28, top=6 at 15.7%, very distributed)
  - my_hand_count: my hand size (nunique=35, NOT currently in model)

op_hand_count captures opponent's resource state:
  - Low hand (1-3): opponent is in trouble, limited options → urgency differs
  - High hand (7+): opponent has many cards, dangerous → different strategic priority
  This is a direct signal about opponent's threat level.

my_hand_count captures my resource state:
  - Low hand: limited options this turn, need to play carefully
  - High hand: many options, can plan ahead

Both are decision-level features (same for all options in a decision) and are
computed directly from game state (NOT from is_chosen) → no target leakage risk.

Verified in UNKNOWN_0 context:
  op_hand_count: nunique=28, top=6 (15.7%), mean=7.49 — excellent distribution
  my_hand_count: in alakazam_option_model_df, nunique=35

NOTE: my_hand_count might already be in the LGBM base features. The
_extra_numeric_avail check ensures no-op if already included.

Based on v08d34 (truncation_level=1, position_winrate, top1=0.5485 BEST).

Changes from v08d34:
  1. EXPERIMENT_NAME → v0_08d39_hand_counts
  2. RUN_PREFIX → pokemon-20260627-v0-08d39
  3. Cell[7]: Preload from v08d19 (same split)
  4. Cell[19]: Add op_hand_count, my_hand_count to _extra_numeric_candidates
"""
import json

SRC = '/kaggle/working/pokemon-20260627-v0-08d34.ipynb'
DST = '/kaggle/working/pokemon-20260627-v0-08d39.ipynb'

with open(SRC) as f:
    nb = json.load(f)

cells = nb['cells']

# ── Cell[1]: Config ───────────────────────────────────────────────────────────
src1 = ''.join(cells[1]['source'])

OLD_EXP = "EXPERIMENT_NAME = 'v0_08d34_trunc1'"
NEW_EXP = "EXPERIMENT_NAME = 'v0_08d39_hand_counts'"
assert OLD_EXP in src1
src1 = src1.replace(OLD_EXP, NEW_EXP)

OLD_PFX = "RUN_PREFIX = 'pokemon-20260627-v0-08d34'"
NEW_PFX = "RUN_PREFIX = 'pokemon-20260627-v0-08d39'"
assert OLD_PFX in src1
src1 = src1.replace(OLD_PFX, NEW_PFX)

cells[1]['source'] = src1.splitlines(keepends=True)

# ── Cell[7]: Preload ──────────────────────────────────────────────────────────
src7 = ''.join(cells[7]['source'])

OLD_PRELOAD = "# v08d34: preload from v08d19 (same split)\n"
NEW_PRELOAD = "# v08d39: preload from v08d19 (same split)\n"
assert OLD_PRELOAD in src7
src7 = src7.replace(OLD_PRELOAD, NEW_PRELOAD)

OLD_PRINT = "    print(f'v08d34 preload: {len(DECISION_ROWS_DF)} decisions from {_pfx}')\n"
NEW_PRINT = "    print(f'v08d39 preload: {len(DECISION_ROWS_DF)} decisions from {_pfx}')\n"
assert OLD_PRINT in src7
src7 = src7.replace(OLD_PRINT, NEW_PRINT)

OLD_FAIL = "    print('v08d34 preload: no cache found, mining from scratch')\n"
NEW_FAIL = "    print('v08d39 preload: no cache found, mining from scratch')\n"
assert OLD_FAIL in src7
src7 = src7.replace(OLD_FAIL, NEW_FAIL)

cells[7]['source'] = src7.splitlines(keepends=True)

# ── Cell[19]: Add op_hand_count + my_hand_count ───────────────────────────────
src19 = ''.join(cells[19]['source'])

OLD_CANDS = (
    "        _extra_numeric_candidates = ['supporter_played', 'stadium_played', 'energy_attached',\n"
    "                                      'my_discard_count', 'op_discard_count', 'prize_gap', 'steps_since_op',\n"
    "                                      'position_winrate']\n"
)
NEW_CANDS = (
    "        # v08d39: hand counts — decision-level, not derived from is_chosen, good variance\n"
    "        _extra_numeric_candidates = ['supporter_played', 'stadium_played', 'energy_attached',\n"
    "                                      'my_discard_count', 'op_discard_count', 'prize_gap', 'steps_since_op',\n"
    "                                      'position_winrate', 'op_hand_count', 'my_hand_count']\n"
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

assert 'v0_08d39_hand_counts' in s1
assert "RUN_PREFIX = 'pokemon-20260627-v0-08d39'" in s1
assert 'v08d19' in s7
assert 'op_hand_count' in s19
assert 'my_hand_count' in s19
assert 'lambdarank_truncation_level' in s19
assert "lambdarank_truncation_level': 1" in s19
assert 'position_winrate' in s19
assert 'op_last_context' in s19
assert '_winner_weight = 4.0' in s19
full = ''.join(''.join(c['source']) for c in nb2['cells'])
assert 'lambdarank' in full
print('All sanity checks passed.')
