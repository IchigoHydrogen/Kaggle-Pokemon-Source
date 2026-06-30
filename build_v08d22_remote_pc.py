"""Build pokemon-20260627-v0-08d22-remote-pc.ipynb from v08d19.

v08d22: winner_weight tuning — 4x → 8x.

op_last_context was the breakthrough (+0.0229). Capacity/data scaling didn't help.
Next direction: increase winner emphasis in LambdaRank training.

With winner_weight=8x, LGBM penalizes mistakes on winning games more heavily.
Hypothesis: the model should bias more strongly toward options chosen by WINNERS,
which should improve inference (we ARE trying to win).

Current state (v08d19 baseline):
  winner_weight=4x → top1=0.5469

Changes from v08d19:
  1. EXPERIMENT_NAME → v0_08d22_wt8
  2. RUN_PREFIX → pokemon-20260627-v0-08d22
  3. Cell[7]: Preload from v08d19
  4. Cell[19]: winner_weight 4 → 8
  5. Cell[1]: Keep 350k rows (same as v08d19)
"""
import json

SRC = '/kaggle/working/pokemon-20260627-v0-08d19.ipynb'
DST = '/kaggle/working/pokemon-20260627-v0-08d22.ipynb'

with open(SRC) as f:
    nb = json.load(f)

cells = nb['cells']

# ── Cell[1]: Config ───────────────────────────────────────────────────────────
src1 = ''.join(cells[1]['source'])

OLD_EXP = "EXPERIMENT_NAME = 'v0_08d19_op_seq'"
NEW_EXP = "EXPERIMENT_NAME = 'v0_08d22_wt8'"
assert OLD_EXP in src1
src1 = src1.replace(OLD_EXP, NEW_EXP)

OLD_PFX = "RUN_PREFIX = 'pokemon-20260627-v0-08d19'"
NEW_PFX = "RUN_PREFIX = 'pokemon-20260627-v0-08d22'"
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
    "# v08d22: preload from v08d19 (same split)\n"
    "_PRELOAD_CANDIDATES = [\n"
    "    Path('/kaggle/working/pokemon-20260627-v0-08d19/pokemon-20260627-v0-08d19'),\n"
    "    Path('/kaggle/working/pokemon-20260627-v0-08d18/pokemon-20260627-v0-08d18'),\n"
    "]\n"
)
assert OLD_PRELOAD in src7
src7 = src7.replace(OLD_PRELOAD, NEW_PRELOAD)

OLD_PRINT = "    print(f'v08d19 preload: {len(DECISION_ROWS_DF)} decisions from {_pfx}')\n"
NEW_PRINT = "    print(f'v08d22 preload: {len(DECISION_ROWS_DF)} decisions from {_pfx}')\n"
assert OLD_PRINT in src7
src7 = src7.replace(OLD_PRINT, NEW_PRINT)

OLD_FAIL = "    print('v08d19 preload: v08d18 cache not found, mining from scratch')\n"
NEW_FAIL = "    print('v08d22 preload: no cache found, mining from scratch')\n"
assert OLD_FAIL in src7
src7 = src7.replace(OLD_FAIL, NEW_FAIL)

cells[7]['source'] = src7.splitlines(keepends=True)

# ── Cell[19]: winner_weight 4 → 8 ─────────────────────────────────────────────
src19 = ''.join(cells[19]['source'])

# Find winner_weight setting
OLD_WT = "_winner_weight = 4.0"
NEW_WT = "_winner_weight = 8.0"
assert OLD_WT in src19, f'_winner_weight not found'
src19 = src19.replace(OLD_WT, NEW_WT)

OLD_WTL = "winner_wt4"
NEW_WTL = "winner_wt8"
src19 = src19.replace(OLD_WTL, NEW_WTL)

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

assert 'v0_08d22_wt8' in s1
assert "RUN_PREFIX = 'pokemon-20260627-v0-08d22'" in s1
assert 'v08d19' in s7, 'preload from v08d19'
assert '_winner_weight = 8.0' in s19, 'winner_weight=8'
assert '_winner_weight = 4.0' not in s19, 'old winner_weight still present'
assert 'op_last_context' in s19, 'op_last_context kept'
assert 'prize_gap' in s19, 'prize_gap kept'
full = ''.join(''.join(c['source']) for c in nb2['cells'])
assert 'lambdarank' in full
print('All sanity checks passed.')
