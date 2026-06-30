"""Build pokemon-20260627-v0-08d24-remote-pc.ipynb from v08d19.

v08d24: MLP reranker submission (first Kaggle test of MLP).

KEY OBSERVATION: MLP has consistently achieved top1=0.5596~0.5622 across v08d19-v08d23,
outperforming LGBM (v08d19 best: 0.5469) by +0.015~+0.03.
Despite this, MLP has never been submitted to Kaggle.
This is an unused trained asset ("怖い" reason, not technical impossibility).

Changes from v08d19:
  1. EXPERIMENT_NAME → v0_08d24_mlp_sub
  2. RUN_PREFIX → pokemon-20260627-v0-08d24
  3. Cell[7]: Preload from v08d19 (same split)
  4. Cell[1]: USE_MLP_RERANKER_FOR_SUBMISSION = True
  5. Cell[1]: MLP_SUBMISSION_CONTEXTS = ['UNKNOWN_0']
  (LGBM training unchanged; submission variant uses MLP reranker on top)
"""
import json

SRC = '/kaggle/working/pokemon-20260627-v0-08d19.ipynb'
DST = '/kaggle/working/pokemon-20260627-v0-08d24.ipynb'

with open(SRC) as f:
    nb = json.load(f)

cells = nb['cells']

# ── Cell[1]: Config ───────────────────────────────────────────────────────────
src1 = ''.join(cells[1]['source'])

OLD_EXP = "EXPERIMENT_NAME = 'v0_08d19_op_seq'"
NEW_EXP = "EXPERIMENT_NAME = 'v0_08d24_mlp_sub'"
assert OLD_EXP in src1
src1 = src1.replace(OLD_EXP, NEW_EXP)

OLD_PFX = "RUN_PREFIX = 'pokemon-20260627-v0-08d19'"
NEW_PFX = "RUN_PREFIX = 'pokemon-20260627-v0-08d24'"
assert OLD_PFX in src1
src1 = src1.replace(OLD_PFX, NEW_PFX)

OLD_MLP_FLAG = "USE_MLP_RERANKER_FOR_SUBMISSION = False"
NEW_MLP_FLAG = "USE_MLP_RERANKER_FOR_SUBMISSION = True"
assert OLD_MLP_FLAG in src1, f'USE_MLP_RERANKER_FOR_SUBMISSION not found'
src1 = src1.replace(OLD_MLP_FLAG, NEW_MLP_FLAG)

OLD_CTX = "MLP_SUBMISSION_CONTEXTS = []"
NEW_CTX = "MLP_SUBMISSION_CONTEXTS = ['UNKNOWN_0']"
assert OLD_CTX in src1, f'MLP_SUBMISSION_CONTEXTS not found'
src1 = src1.replace(OLD_CTX, NEW_CTX)

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
    "# v08d24: preload from v08d19 (same split)\n"
    "_PRELOAD_CANDIDATES = [\n"
    "    Path('/kaggle/working/pokemon-20260627-v0-08d19/pokemon-20260627-v0-08d19'),\n"
    "    Path('/kaggle/working/pokemon-20260627-v0-08d18/pokemon-20260627-v0-08d18'),\n"
    "]\n"
)
assert OLD_PRELOAD in src7
src7 = src7.replace(OLD_PRELOAD, NEW_PRELOAD)

OLD_PRINT = "    print(f'v08d19 preload: {len(DECISION_ROWS_DF)} decisions from {_pfx}')\n"
NEW_PRINT = "    print(f'v08d24 preload: {len(DECISION_ROWS_DF)} decisions from {_pfx}')\n"
assert OLD_PRINT in src7
src7 = src7.replace(OLD_PRINT, NEW_PRINT)

OLD_FAIL = "    print('v08d19 preload: v08d18 cache not found, mining from scratch')\n"
NEW_FAIL = "    print('v08d24 preload: no cache found, mining from scratch')\n"
assert OLD_FAIL in src7
src7 = src7.replace(OLD_FAIL, NEW_FAIL)

cells[7]['source'] = src7.splitlines(keepends=True)

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

assert 'v0_08d24_mlp_sub' in s1
assert "RUN_PREFIX = 'pokemon-20260627-v0-08d24'" in s1
assert 'USE_MLP_RERANKER_FOR_SUBMISSION = True' in s1
assert "MLP_SUBMISSION_CONTEXTS = ['UNKNOWN_0']" in s1
assert 'v08d19' in s7, 'preload from v08d19'
assert 'op_last_context' in s19, 'op_last_context kept'
assert 'prize_gap' in s19, 'prize_gap kept'
assert '_winner_weight = 4.0' in s19, 'winner_weight=4x kept'
full = ''.join(''.join(c['source']) for c in nb2['cells'])
assert 'lambdarank' in full
print('All sanity checks passed.')
