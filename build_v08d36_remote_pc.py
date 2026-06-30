"""Build pokemon-20260627-v0-08d36-remote-pc.ipynb from v08d34.

v08d36: winner_weight=8.0 × lambdarank_truncation_level=1.

v08d34 established: truncation_level=1 is the key structural change (top1=0.5485, NEW BEST).
v08d35 confirmed: turn_action_count hurts top1 even with truncation_level=1.

NEXT LEVER: winner_weight (currently 4.0).

With truncation_level=1, ALL gradient computation focuses on:
  "Does winner player's #1 choice rank above unchosen options?"

Increasing winner_weight from 4.0 to 8.0 doubles the emphasis on winner decisions:
  - Loser decisions weight=1.0 (unchanged)
  - Winner decisions weight=8.0 (doubled from 4.0)

Combined effect of truncation_level=1 + winner_weight=8.0:
  The model maximally optimizes: "get the #1 choice right for WINNING game states"
  This mirrors the competition's implicit signal — winning players made better decisions.

Prior winner_weight experiments (v08d18-v08d19 era established winner_weight=4.0 as optimal
over 2.0 and 3.0). We haven't tested 8.0. With truncation_level=1, the interaction
may differ — focused gradient + amplified winner signal could be synergistic.

Risk: overfitting to winner game states. Validation includes both winner and loser
decisions — very high winner_weight might underfit loser-state decisions.

Based on v08d34 (truncation_level=1, position_winrate, top1=0.5485 BEST).

Changes from v08d34:
  1. EXPERIMENT_NAME → v0_08d36_wt8_trunc1
  2. RUN_PREFIX → pokemon-20260627-v0-08d36
  3. Cell[7]: Preload from v08d19 (same split)
  4. Cell[19]: winner_weight 4.0 → 8.0
"""
import json

SRC = '/kaggle/working/pokemon-20260627-v0-08d34.ipynb'
DST = '/kaggle/working/pokemon-20260627-v0-08d36.ipynb'

with open(SRC) as f:
    nb = json.load(f)

cells = nb['cells']

# ── Cell[1]: Config ───────────────────────────────────────────────────────────
src1 = ''.join(cells[1]['source'])

OLD_EXP = "EXPERIMENT_NAME = 'v0_08d34_trunc1'"
NEW_EXP = "EXPERIMENT_NAME = 'v0_08d36_wt8_trunc1'"
assert OLD_EXP in src1
src1 = src1.replace(OLD_EXP, NEW_EXP)

OLD_PFX = "RUN_PREFIX = 'pokemon-20260627-v0-08d34'"
NEW_PFX = "RUN_PREFIX = 'pokemon-20260627-v0-08d36'"
assert OLD_PFX in src1
src1 = src1.replace(OLD_PFX, NEW_PFX)

cells[1]['source'] = src1.splitlines(keepends=True)

# ── Cell[7]: Preload ──────────────────────────────────────────────────────────
src7 = ''.join(cells[7]['source'])

OLD_PRELOAD = "# v08d34: preload from v08d19 (same split)\n"
NEW_PRELOAD = "# v08d36: preload from v08d19 (same split)\n"
assert OLD_PRELOAD in src7
src7 = src7.replace(OLD_PRELOAD, NEW_PRELOAD)

OLD_PRINT = "    print(f'v08d34 preload: {len(DECISION_ROWS_DF)} decisions from {_pfx}')\n"
NEW_PRINT = "    print(f'v08d36 preload: {len(DECISION_ROWS_DF)} decisions from {_pfx}')\n"
assert OLD_PRINT in src7
src7 = src7.replace(OLD_PRINT, NEW_PRINT)

OLD_FAIL = "    print('v08d34 preload: no cache found, mining from scratch')\n"
NEW_FAIL = "    print('v08d36 preload: no cache found, mining from scratch')\n"
assert OLD_FAIL in src7
src7 = src7.replace(OLD_FAIL, NEW_FAIL)

cells[7]['source'] = src7.splitlines(keepends=True)

# ── Cell[19]: winner_weight 4.0 → 8.0 ────────────────────────────────────────
src19 = ''.join(cells[19]['source'])

OLD_WT = "                _winner_weight = 4.0\n"
NEW_WT = "                _winner_weight = 8.0  # v08d36: doubled from 4.0, combined with trunc1\n"
assert OLD_WT in src19, 'winner_weight=4.0 not found in Cell[19]'
src19 = src19.replace(OLD_WT, NEW_WT)

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

assert 'v0_08d36_wt8_trunc1' in s1
assert "RUN_PREFIX = 'pokemon-20260627-v0-08d36'" in s1
assert 'v08d19' in s7
assert '_winner_weight = 8.0' in s19
assert '_winner_weight = 4.0' not in s19, 'old winner_weight=4.0 must be gone'
assert 'lambdarank_truncation_level' in s19
assert "lambdarank_truncation_level': 1" in s19
assert 'position_winrate' in s19
assert 'op_last_context' in s19
full = ''.join(''.join(c['source']) for c in nb2['cells'])
assert 'lambdarank' in full
print('All sanity checks passed.')
