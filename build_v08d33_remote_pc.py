"""Build pokemon-20260627-v0-08d33-remote-pc.ipynb from v08d28.

v08d33: turn_action_count ONLY — single-feature ablation.

v08d32 showed 8 new features, but ONLY turn_action_count entered top10 (44142 importance).
The 7 other features (evolution state, bench counts, etc.) had near-zero importance.
turn_action_count absorbed importance from num_options (175k→109k).

This experiment isolates turn_action_count:
  - ONLY add turn_action_count to _extra_numeric_candidates
  - Keep everything else from v08d28 (position_winrate, winner_weight=4x)
  - If top1 improves: turn_action_count is genuinely useful
  - If top1 stays same: the feature is noise for top1 despite AUC gain

turn_action_count (nunique=31, top=1 at 12.9%, mean=6.85):
  "How many actions have been taken THIS turn?"
  - Low value (1): early in turn, more options open (supporter not yet played)
  - High value (6+): late in turn, most cards already played
  - Captures INTRA-TURN PHASE — different decisions are correct early vs late in turn

Hypothesis: early in a turn (action=1), the decision "play supporter now" is different
from later in the turn. The model currently cannot distinguish "should I play this card
now vs later in the same turn" because it doesn't know where we are in the turn.

No inference proxy needed — turn_action_count is a directly observable game state feature
already in ALAKAZAM_OPTION_MODEL_DF (same code path at train and inference time).

Based on v08d28 (includes position_winrate, +0.0008).

Changes from v08d28:
  1. EXPERIMENT_NAME → v0_08d33_tac_only
  2. RUN_PREFIX → pokemon-20260627-v0-08d33
  3. Cell[7]: Preload from v08d19 (same split)
  4. Cell[19]: Add ONLY turn_action_count to _extra_numeric_candidates
"""
import json

SRC = '/kaggle/working/pokemon-20260627-v0-08d28.ipynb'
DST = '/kaggle/working/pokemon-20260627-v0-08d33.ipynb'

with open(SRC) as f:
    nb = json.load(f)

cells = nb['cells']

# ── Cell[1]: Config ───────────────────────────────────────────────────────────
src1 = ''.join(cells[1]['source'])

OLD_EXP = "EXPERIMENT_NAME = 'v0_08d28_winrate_feat'"
NEW_EXP = "EXPERIMENT_NAME = 'v0_08d33_tac_only'"
assert OLD_EXP in src1
src1 = src1.replace(OLD_EXP, NEW_EXP)

OLD_PFX = "RUN_PREFIX = 'pokemon-20260627-v0-08d28'"
NEW_PFX = "RUN_PREFIX = 'pokemon-20260627-v0-08d33'"
assert OLD_PFX in src1
src1 = src1.replace(OLD_PFX, NEW_PFX)

cells[1]['source'] = src1.splitlines(keepends=True)

# ── Cell[7]: Preload ──────────────────────────────────────────────────────────
src7 = ''.join(cells[7]['source'])

OLD_PRELOAD = "# v08d28: preload from v08d19 (same split)\n"
NEW_PRELOAD = "# v08d33: preload from v08d19 (same split)\n"
assert OLD_PRELOAD in src7
src7 = src7.replace(OLD_PRELOAD, NEW_PRELOAD)

OLD_PRINT = "    print(f'v08d28 preload: {len(DECISION_ROWS_DF)} decisions from {_pfx}')\n"
NEW_PRINT = "    print(f'v08d33 preload: {len(DECISION_ROWS_DF)} decisions from {_pfx}')\n"
assert OLD_PRINT in src7
src7 = src7.replace(OLD_PRINT, NEW_PRINT)

OLD_FAIL = "    print('v08d28 preload: no cache found, mining from scratch')\n"
NEW_FAIL = "    print('v08d33 preload: no cache found, mining from scratch')\n"
assert OLD_FAIL in src7
src7 = src7.replace(OLD_FAIL, NEW_FAIL)

cells[7]['source'] = src7.splitlines(keepends=True)

# ── Cell[19]: Add ONLY turn_action_count ─────────────────────────────────────
src19 = ''.join(cells[19]['source'])

OLD_CANDS = (
    "        _extra_numeric_candidates = ['supporter_played', 'stadium_played', 'energy_attached',\n"
    "                                      'my_discard_count', 'op_discard_count', 'prize_gap', 'steps_since_op',\n"
    "                                      'position_winrate']\n"
)
NEW_CANDS = (
    "        # v08d33: turn_action_count single-feature ablation\n"
    "        # Only new feature from v08d32 that had significant importance (44142, #6)\n"
    "        # Captures intra-turn phase: early actions vs late in the same turn\n"
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

assert 'v0_08d33_tac_only' in s1
assert "RUN_PREFIX = 'pokemon-20260627-v0-08d33'" in s1
assert 'v08d19' in s7
assert 'turn_action_count' in s19
# Check the candidates list specifically (not the whole cell which may have comments)
_cands_line = [l for l in s19.splitlines() if '_extra_numeric_candidates = [' in l or ("'turn_action_count'" in l and 'position_winrate' in l)]
assert any("'turn_action_count'" in l for l in s19.splitlines() if '_extra_numeric_candidates' in s19), 'turn_action_count not in candidates'
assert 'position_winrate' in s19, 'position_winrate from v08d28 must be kept'
assert 'op_last_context' in s19
assert '_winner_weight = 4.0' in s19
full = ''.join(''.join(c['source']) for c in nb2['cells'])
assert 'lambdarank' in full
print('All sanity checks passed.')
