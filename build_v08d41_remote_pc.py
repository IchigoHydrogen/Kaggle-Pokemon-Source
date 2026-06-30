"""Build pokemon-20260627-v0-08d41-remote-pc.ipynb from v08d34.

v08d41: learning_rate=0.02 (halved from 0.05) on trunc1 base.

CONFIRMED:
  - truncation_level=1 is the key structural improvement (top1=0.5485)
  - Feature additions beyond position_winrate: no help
  - winner_weight>4.0: overfits
  - max_rows>350k: hurts (adds noisy options, same decisions)

NEXT: Learning rate tuning.
Current: lr=0.05, best_iter=136 → effective learning ≈ 6.8 "units"
Proposed: lr=0.02, ~340 iterations for same effective learning, but may find finer minimum.

With truncation_level=1 focusing all gradient on top-1:
  - lr=0.05 might overshoot the optimal weight values
  - lr=0.02 allows more precise gradient steps toward the true minimum
  - Best_iter should roughly double (136 → ~272), still within time budget

This is a hyperparameter refinement, not a structural change.

Based on v08d34 (truncation_level=1, position_winrate, top1=0.5485 BEST).

Changes from v08d34:
  1. EXPERIMENT_NAME → v0_08d41_lr002
  2. RUN_PREFIX → pokemon-20260627-v0-08d41
  3. Cell[7]: Preload from v08d19 (same split)
  4. Cell[19]: learning_rate 0.05 → 0.02
"""
import json

SRC = '/kaggle/working/pokemon-20260627-v0-08d34.ipynb'
DST = '/kaggle/working/pokemon-20260627-v0-08d41.ipynb'

with open(SRC) as f:
    nb = json.load(f)

cells = nb['cells']

# ── Cell[1]: Config ───────────────────────────────────────────────────────────
src1 = ''.join(cells[1]['source'])

OLD_EXP = "EXPERIMENT_NAME = 'v0_08d34_trunc1'"
NEW_EXP = "EXPERIMENT_NAME = 'v0_08d41_lr002'"
assert OLD_EXP in src1
src1 = src1.replace(OLD_EXP, NEW_EXP)

OLD_PFX = "RUN_PREFIX = 'pokemon-20260627-v0-08d34'"
NEW_PFX = "RUN_PREFIX = 'pokemon-20260627-v0-08d41'"
assert OLD_PFX in src1
src1 = src1.replace(OLD_PFX, NEW_PFX)

cells[1]['source'] = src1.splitlines(keepends=True)

# ── Cell[7]: Preload ──────────────────────────────────────────────────────────
src7 = ''.join(cells[7]['source'])

OLD_PRELOAD = "# v08d34: preload from v08d19 (same split)\n"
NEW_PRELOAD = "# v08d41: preload from v08d19 (same split)\n"
assert OLD_PRELOAD in src7
src7 = src7.replace(OLD_PRELOAD, NEW_PRELOAD)

OLD_PRINT = "    print(f'v08d34 preload: {len(DECISION_ROWS_DF)} decisions from {_pfx}')\n"
NEW_PRINT = "    print(f'v08d41 preload: {len(DECISION_ROWS_DF)} decisions from {_pfx}')\n"
assert OLD_PRINT in src7
src7 = src7.replace(OLD_PRINT, NEW_PRINT)

OLD_FAIL = "    print('v08d34 preload: no cache found, mining from scratch')\n"
NEW_FAIL = "    print('v08d41 preload: no cache found, mining from scratch')\n"
assert OLD_FAIL in src7
src7 = src7.replace(OLD_FAIL, NEW_FAIL)

cells[7]['source'] = src7.splitlines(keepends=True)

# ── Cell[19]: learning_rate 0.05 → 0.02 ──────────────────────────────────────
src19 = ''.join(cells[19]['source'])

OLD_LR = "                    'learning_rate': 0.05,\n"
NEW_LR = "                    'learning_rate': 0.02,  # v08d41: halved from 0.05\n"
assert OLD_LR in src19, 'learning_rate=0.05 not found in Cell[19]'
src19 = src19.replace(OLD_LR, NEW_LR)

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

assert 'v0_08d41_lr002' in s1
assert "RUN_PREFIX = 'pokemon-20260627-v0-08d41'" in s1
assert 'v08d19' in s7
assert "'learning_rate': 0.02" in s19
assert "'learning_rate': 0.05" not in s19, 'old lr=0.05 must be gone'
assert 'lambdarank_truncation_level' in s19
assert "lambdarank_truncation_level': 1" in s19
assert 'position_winrate' in s19
assert '_winner_weight = 4.0' in s19
full = ''.join(''.join(c['source']) for c in nb2['cells'])
assert 'lambdarank' in full
print('All sanity checks passed.')
