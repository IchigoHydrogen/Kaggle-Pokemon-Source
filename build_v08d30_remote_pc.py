"""Build pokemon-20260627-v0-08d30-remote-pc.ipynb from v08d28.

v08d30: num_leaves=127 (from 63) — model capacity increase.

v08d18-v08d29: 12 experiments. Only v08d19 (+0.019) and v08d28 (+0.0008) improved.
best_iter across experiments: 89-204 with num_leaves=63. This suggests the
model may be underfitting — limited by tree complexity, not by training duration.

HYPOTHESIS: Doubling num_leaves from 63 to 127 allows the model to capture:
  - Higher-order feature interactions (e.g., option_type × prize_gap × archetype)
  - Finer-grained menu position patterns (option_index at higher resolution)
  - More specific option_signature patterns

Based on v08d28 (includes position_winrate, +0.0008).
Same winner_weight=4x training (NOT winners-only — v08d29 showed that fails).

Changes from v08d28:
  1. EXPERIMENT_NAME → v0_08d30_leaves127
  2. RUN_PREFIX → pokemon-20260627-v0-08d30
  3. Cell[7]: Preload from v08d19 (same split)
  4. Cell[19]: num_leaves 63 → 127
"""
import json

SRC = '/kaggle/working/pokemon-20260627-v0-08d28.ipynb'
DST = '/kaggle/working/pokemon-20260627-v0-08d30.ipynb'

with open(SRC) as f:
    nb = json.load(f)

cells = nb['cells']

# ── Cell[1]: Config ───────────────────────────────────────────────────────────
src1 = ''.join(cells[1]['source'])

OLD_EXP = "EXPERIMENT_NAME = 'v0_08d28_winrate_feat'"
NEW_EXP = "EXPERIMENT_NAME = 'v0_08d30_leaves127'"
assert OLD_EXP in src1
src1 = src1.replace(OLD_EXP, NEW_EXP)

OLD_PFX = "RUN_PREFIX = 'pokemon-20260627-v0-08d28'"
NEW_PFX = "RUN_PREFIX = 'pokemon-20260627-v0-08d30'"
assert OLD_PFX in src1
src1 = src1.replace(OLD_PFX, NEW_PFX)

cells[1]['source'] = src1.splitlines(keepends=True)

# ── Cell[7]: Preload from v08d19 ──────────────────────────────────────────────
src7 = ''.join(cells[7]['source'])

OLD_PRELOAD = "# v08d28: preload from v08d19 (same split)\n"
NEW_PRELOAD = "# v08d30: preload from v08d19 (same split)\n"
assert OLD_PRELOAD in src7
src7 = src7.replace(OLD_PRELOAD, NEW_PRELOAD)

OLD_PRINT = "    print(f'v08d28 preload: {len(DECISION_ROWS_DF)} decisions from {_pfx}')\n"
NEW_PRINT = "    print(f'v08d30 preload: {len(DECISION_ROWS_DF)} decisions from {_pfx}')\n"
assert OLD_PRINT in src7
src7 = src7.replace(OLD_PRINT, NEW_PRINT)

OLD_FAIL = "    print('v08d28 preload: no cache found, mining from scratch')\n"
NEW_FAIL = "    print('v08d30 preload: no cache found, mining from scratch')\n"
assert OLD_FAIL in src7
src7 = src7.replace(OLD_FAIL, NEW_FAIL)

cells[7]['source'] = src7.splitlines(keepends=True)

# ── Cell[19]: num_leaves 63 → 127 ────────────────────────────────────────────
src19 = ''.join(cells[19]['source'])

# The num_leaves parameter appears in the LGBM params dict
OLD_LEAVES = "'num_leaves': 63,"
NEW_LEAVES = "'num_leaves': 127,  # v08d30: doubled from 63"
assert OLD_LEAVES in src19, f'num_leaves=63 not found in Cell[19]'
src19 = src19.replace(OLD_LEAVES, NEW_LEAVES)

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

assert 'v0_08d30_leaves127' in s1
assert "RUN_PREFIX = 'pokemon-20260627-v0-08d30'" in s1
assert 'v08d19' in s7
assert "'num_leaves': 127" in s19, 'num_leaves not updated'
assert "'num_leaves': 63," not in s19, 'old num_leaves=63 still present'
assert 'position_winrate' in s19, 'position_winrate from v08d28 must be kept'
assert 'Winner-weighted training' in s19, 'winner weighting must be kept (not winners-only)'
assert 'op_last_context' in s19
full = ''.join(''.join(c['source']) for c in nb2['cells'])
assert 'lambdarank' in full
print('All sanity checks passed.')
