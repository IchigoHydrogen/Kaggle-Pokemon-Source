"""Build pokemon-20260627-v0-08d42-remote-pc.ipynb from v08d28.

v08d42: rank_xendcg objective — alternative to lambdarank.

CONFIRMED BEST CONFIGURATION: v08d34
  lambdarank + truncation_level=1 + position_winrate + lr=0.05 + wt=4.0
  top1=0.5485

Hyperparameter exhaustion:
  - lr: 0.05 optimal (0.02 underfits, 0.08 not tested but likely overfits)
  - winner_weight: 4.0 optimal (8.0 overfits)
  - truncation_level: 1 optimal (5/30 equivalent)
  - num_leaves: 63 optimal (127 no gain)
  - max_rows: 350k optimal (420k hurts)
  - features: exhausted (all informative features already in model)

STRUCTURAL CHANGE: rank_xendcg objective.

LightGBM supports two ranking objectives:
  1. lambdarank: approximates NDCG gradient via pairwise lambda weighting
  2. rank_xendcg: direct cross-entropy optimization with NDCG-like weights
     (XE-NDCG: Bruch et al. 2019 - "An Analysis of the Softmax Cross Entropy Loss
     for Learning-to-Rank with Binary Relevance")

rank_xendcg advantages:
  - Optimizes a proper upper bound on NDCG loss (theoretical guarantee)
  - More stable gradients (entropy-based, not pairwise)
  - Better calibrated scores (cross-entropy normalization)
  - Might work differently with the winner_weight scheme

NOTE: lambdarank_truncation_level does NOT apply to rank_xendcg.
This experiment tests rank_xendcg WITHOUT truncation, on v08d28 base.
If rank_xendcg + trunc1 is also possible, that would be v08d43.

Based on v08d28 (position_winrate, no truncation) to isolate the objective change.

Changes from v08d28:
  1. EXPERIMENT_NAME → v0_08d42_xendcg
  2. RUN_PREFIX → pokemon-20260627-v0-08d42
  3. Cell[7]: Preload from v08d19 (same split)
  4. Cell[19]: objective 'lambdarank' → 'rank_xendcg'
  5. Cell[19]: Remove lambdarank_truncation_level (not applicable to rank_xendcg)
"""
import json

SRC = '/kaggle/working/pokemon-20260627-v0-08d28.ipynb'
DST = '/kaggle/working/pokemon-20260627-v0-08d42.ipynb'

with open(SRC) as f:
    nb = json.load(f)

cells = nb['cells']

# ── Cell[1]: Config ───────────────────────────────────────────────────────────
src1 = ''.join(cells[1]['source'])

OLD_EXP = "EXPERIMENT_NAME = 'v0_08d28_winrate_feat'"
NEW_EXP = "EXPERIMENT_NAME = 'v0_08d42_xendcg'"
assert OLD_EXP in src1
src1 = src1.replace(OLD_EXP, NEW_EXP)

OLD_PFX = "RUN_PREFIX = 'pokemon-20260627-v0-08d28'"
NEW_PFX = "RUN_PREFIX = 'pokemon-20260627-v0-08d42'"
assert OLD_PFX in src1
src1 = src1.replace(OLD_PFX, NEW_PFX)

cells[1]['source'] = src1.splitlines(keepends=True)

# ── Cell[7]: Preload ──────────────────────────────────────────────────────────
src7 = ''.join(cells[7]['source'])

OLD_PRELOAD = "# v08d28: preload from v08d19 (same split)\n"
NEW_PRELOAD = "# v08d42: preload from v08d19 (same split)\n"
assert OLD_PRELOAD in src7
src7 = src7.replace(OLD_PRELOAD, NEW_PRELOAD)

OLD_PRINT = "    print(f'v08d28 preload: {len(DECISION_ROWS_DF)} decisions from {_pfx}')\n"
NEW_PRINT = "    print(f'v08d42 preload: {len(DECISION_ROWS_DF)} decisions from {_pfx}')\n"
assert OLD_PRINT in src7
src7 = src7.replace(OLD_PRINT, NEW_PRINT)

OLD_FAIL = "    print('v08d28 preload: no cache found, mining from scratch')\n"
NEW_FAIL = "    print('v08d42 preload: no cache found, mining from scratch')\n"
assert OLD_FAIL in src7
src7 = src7.replace(OLD_FAIL, NEW_FAIL)

cells[7]['source'] = src7.splitlines(keepends=True)

# ── Cell[19]: objective lambdarank → rank_xendcg ─────────────────────────────
src19 = ''.join(cells[19]['source'])

OLD_OBJ = "                    'objective': 'lambdarank',\n"
NEW_OBJ = "                    'objective': 'rank_xendcg',  # v08d42: XE-NDCG (Bruch 2019)\n"
assert OLD_OBJ in src19, 'lambdarank objective not found in Cell[19]'
src19 = src19.replace(OLD_OBJ, NEW_OBJ)

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

assert 'v0_08d42_xendcg' in s1
assert "RUN_PREFIX = 'pokemon-20260627-v0-08d42'" in s1
assert 'v08d19' in s7
assert 'rank_xendcg' in s19
assert "'objective': 'lambdarank'" not in s19, 'old lambdarank objective must be gone'
assert 'position_winrate' in s19
assert '_winner_weight = 4.0' in s19
assert 'op_last_context' in s19
full = ''.join(''.join(c['source']) for c in nb2['cells'])
assert 'rank_xendcg' in full
print('All sanity checks passed.')
