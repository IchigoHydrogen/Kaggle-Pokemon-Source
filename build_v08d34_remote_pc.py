"""Build pokemon-20260627-v0-08d34-remote-pc.ipynb from v08d28.

v08d34: lambdarank_truncation_level=1 — directly optimize top-1 selection.

PATTERN OBSERVED across v08d19-v08d33:
  - Feature additions: AUC improves, top1 unchanged or slightly down
  - Training changes: mostly fail
  - ONLY value signals (position_winrate) and temporal context improved top1

ROOT CAUSE HYPOTHESIS:
  LambdaRank with truncation_level=30 (default) spreads gradient signal across
  all 30 positions. This improves overall ranking (AUC) but diffuses the signal
  away from the critical top-1 decision.

FIX: lambdarank_truncation_level=1 focuses ALL gradient computation on:
  "Does my #1 ranked option match the expert's chosen option?"

This directly maximizes: P(rank(chosen_option) = 1) ≡ top1 accuracy.

LightGBM docs:
  lambdarank_truncation_level: optimal truncation point for early stop.
  A smaller value focuses lambdas more aggressively on top positions.
  Default=30.

Expected behavior:
  - best_iter might change (model optimizes different objective)
  - top1 should improve (directly optimized)
  - AUC/top3 might decrease (no longer optimized)
  - This is the RIGHT trade-off: competition evaluates top1, not AUC

Based on v08d28 (includes position_winrate, +0.0008). NOT v08d32/33 since
turn_action_count hurt top1 (-0.0007).

Changes from v08d28:
  1. EXPERIMENT_NAME → v0_08d34_trunc1
  2. RUN_PREFIX → pokemon-20260627-v0-08d34
  3. Cell[7]: Preload from v08d19 (same split)
  4. Cell[19]: Add lambdarank_truncation_level=1 to LGBM params
"""
import json

SRC = '/kaggle/working/pokemon-20260627-v0-08d28.ipynb'
DST = '/kaggle/working/pokemon-20260627-v0-08d34.ipynb'

with open(SRC) as f:
    nb = json.load(f)

cells = nb['cells']

# ── Cell[1]: Config ───────────────────────────────────────────────────────────
src1 = ''.join(cells[1]['source'])

OLD_EXP = "EXPERIMENT_NAME = 'v0_08d28_winrate_feat'"
NEW_EXP = "EXPERIMENT_NAME = 'v0_08d34_trunc1'"
assert OLD_EXP in src1
src1 = src1.replace(OLD_EXP, NEW_EXP)

OLD_PFX = "RUN_PREFIX = 'pokemon-20260627-v0-08d28'"
NEW_PFX = "RUN_PREFIX = 'pokemon-20260627-v0-08d34'"
assert OLD_PFX in src1
src1 = src1.replace(OLD_PFX, NEW_PFX)

cells[1]['source'] = src1.splitlines(keepends=True)

# ── Cell[7]: Preload ──────────────────────────────────────────────────────────
src7 = ''.join(cells[7]['source'])

OLD_PRELOAD = "# v08d28: preload from v08d19 (same split)\n"
NEW_PRELOAD = "# v08d34: preload from v08d19 (same split)\n"
assert OLD_PRELOAD in src7
src7 = src7.replace(OLD_PRELOAD, NEW_PRELOAD)

OLD_PRINT = "    print(f'v08d28 preload: {len(DECISION_ROWS_DF)} decisions from {_pfx}')\n"
NEW_PRINT = "    print(f'v08d34 preload: {len(DECISION_ROWS_DF)} decisions from {_pfx}')\n"
assert OLD_PRINT in src7
src7 = src7.replace(OLD_PRINT, NEW_PRINT)

OLD_FAIL = "    print('v08d28 preload: no cache found, mining from scratch')\n"
NEW_FAIL = "    print('v08d34 preload: no cache found, mining from scratch')\n"
assert OLD_FAIL in src7
src7 = src7.replace(OLD_FAIL, NEW_FAIL)

cells[7]['source'] = src7.splitlines(keepends=True)

# ── Cell[19]: Add lambdarank_truncation_level=1 ───────────────────────────────
src19 = ''.join(cells[19]['source'])

OLD_PARAMS = (
    "                   'num_leaves': 63,\n"
    "                    'min_child_samples': 10,\n"
)
NEW_PARAMS = (
    "                   'num_leaves': 63,\n"
    "                    'lambdarank_truncation_level': 1,  # v08d34: focus gradients on top-1\n"
    "                    'min_child_samples': 10,\n"
)
assert OLD_PARAMS in src19, 'num_leaves/lr params not found in Cell[19]'
src19 = src19.replace(OLD_PARAMS, NEW_PARAMS)

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

assert 'v0_08d34_trunc1' in s1
assert "RUN_PREFIX = 'pokemon-20260627-v0-08d34'" in s1
assert 'v08d19' in s7
assert 'lambdarank_truncation_level' in s19
assert "lambdarank_truncation_level': 1" in s19
assert 'position_winrate' in s19, 'position_winrate from v08d28 must be kept'
assert 'op_last_context' in s19
assert '_winner_weight = 4.0' in s19
full = ''.join(''.join(c['source']) for c in nb2['cells'])
assert 'lambdarank' in full
print('All sanity checks passed.')
