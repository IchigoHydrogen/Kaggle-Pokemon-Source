"""Build pokemon-20260627-v0-08d38-remote-pc.ipynb from v08d28.

v08d38: lambdarank_truncation_level=5 — sweet-spot search between trunc1 and trunc30.

OBSERVED:
  trunc30 (default, v08d28): top1=0.5477, AUC=0.8701
  trunc1  (v08d34):          top1=0.5485 BEST, AUC=0.8624
  trunc1 + wt8 (v08d36):     top1=0.5385 FAILED (overfits)
  trunc1 + opt_wr (v08d37):  top1=0.5340 FAILED (over-sensitive to leaky feature)

trunc1 improves top1 (+0.0008) at cost of AUC (-0.0077). Can we get MORE top1
improvement with intermediate truncation? trunc5 = "focus on getting top-5 right"
which means: the gradient strongly emphasizes positions 1-5 but doesn't ignore
the full distribution.

HYPOTHESIS: trunc5 might achieve:
  - top1 > trunc30 (more focused gradient than default)
  - top1 >= trunc1 (if 1-5 vs purely 1 focus matters for this task)
  - AUC >= trunc1 (less extreme sacrifice of ranking quality)

This is a hyperparameter sweep point — we have trunc1 and trunc30, and trunc5
is the natural next test.

Also worth noting: trunc5 is less sensitive to leaky features because the gradient
considers positions 1-5 (5 items), giving more signal diversity.

Based on v08d28 (position_winrate, NOT v08d34 to be safe — v08d34's trunc1
is already confirmed, so we're testing the truncation level independently).

Changes from v08d28:
  1. EXPERIMENT_NAME → v0_08d38_trunc5
  2. RUN_PREFIX → pokemon-20260627-v0-08d38
  3. Cell[7]: Preload from v08d19 (same split)
  4. Cell[19]: Add lambdarank_truncation_level=5
"""
import json

SRC = '/kaggle/working/pokemon-20260627-v0-08d28.ipynb'
DST = '/kaggle/working/pokemon-20260627-v0-08d38.ipynb'

with open(SRC) as f:
    nb = json.load(f)

cells = nb['cells']

# ── Cell[1]: Config ───────────────────────────────────────────────────────────
src1 = ''.join(cells[1]['source'])

OLD_EXP = "EXPERIMENT_NAME = 'v0_08d28_winrate_feat'"
NEW_EXP = "EXPERIMENT_NAME = 'v0_08d38_trunc5'"
assert OLD_EXP in src1
src1 = src1.replace(OLD_EXP, NEW_EXP)

OLD_PFX = "RUN_PREFIX = 'pokemon-20260627-v0-08d28'"
NEW_PFX = "RUN_PREFIX = 'pokemon-20260627-v0-08d38'"
assert OLD_PFX in src1
src1 = src1.replace(OLD_PFX, NEW_PFX)

cells[1]['source'] = src1.splitlines(keepends=True)

# ── Cell[7]: Preload ──────────────────────────────────────────────────────────
src7 = ''.join(cells[7]['source'])

OLD_PRELOAD = "# v08d28: preload from v08d19 (same split)\n"
NEW_PRELOAD = "# v08d38: preload from v08d19 (same split)\n"
assert OLD_PRELOAD in src7
src7 = src7.replace(OLD_PRELOAD, NEW_PRELOAD)

OLD_PRINT = "    print(f'v08d28 preload: {len(DECISION_ROWS_DF)} decisions from {_pfx}')\n"
NEW_PRINT = "    print(f'v08d38 preload: {len(DECISION_ROWS_DF)} decisions from {_pfx}')\n"
assert OLD_PRINT in src7
src7 = src7.replace(OLD_PRINT, NEW_PRINT)

OLD_FAIL = "    print('v08d28 preload: no cache found, mining from scratch')\n"
NEW_FAIL = "    print('v08d38 preload: no cache found, mining from scratch')\n"
assert OLD_FAIL in src7
src7 = src7.replace(OLD_FAIL, NEW_FAIL)

cells[7]['source'] = src7.splitlines(keepends=True)

# ── Cell[19]: Add lambdarank_truncation_level=5 ───────────────────────────────
src19 = ''.join(cells[19]['source'])

OLD_PARAMS = (
    "                   'num_leaves': 63,\n"
    "                    'min_child_samples': 10,\n"
)
NEW_PARAMS = (
    "                   'num_leaves': 63,\n"
    "                    'lambdarank_truncation_level': 5,  # v08d38: top-5 focus (trunc1=0.5485, trunc30=0.5477)\n"
    "                    'min_child_samples': 10,\n"
)
assert OLD_PARAMS in src19, 'num_leaves/min_child_samples params not found in Cell[19]'
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

assert 'v0_08d38_trunc5' in s1
assert "RUN_PREFIX = 'pokemon-20260627-v0-08d38'" in s1
assert 'v08d19' in s7
assert 'lambdarank_truncation_level' in s19
assert "lambdarank_truncation_level': 5" in s19
assert 'position_winrate' in s19
assert 'op_last_context' in s19
assert '_winner_weight = 4.0' in s19
full = ''.join(''.join(c['source']) for c in nb2['cells'])
assert 'lambdarank' in full
print('All sanity checks passed.')
