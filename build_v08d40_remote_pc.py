"""Build pokemon-20260627-v0-08d40-remote-pc.ipynb from v08d34.

v08d40: Increase training data cap from 350k to 420k rows.

PATTERN: Feature additions are no-ops (model already has all features from work).
CONFIRMED: truncation_level=1 is the key structural change (v08d34 = best at 0.5485).

NEXT LEVER: Training data quantity.
Current: max_rows = min(350_000, max(50_000, MLP_MAX_ROWS // 2))
       = min(350_000, 350_000) = 350_000

UNKNOWN_0 context has ~411k total option rows in ALAKAZAM_OPTION_MODEL_DF.
We're leaving ~61k rows (6k decisions) unused — 17% of available data.

Fix: change the hard-coded 350k cap to 420k, which allows all 411k rows to be used.
    max_rows = min(420_000, max(50_000, MLP_MAX_ROWS))
              = min(420_000, 700_000) = 420_000

More training data (+17%) with same validation split (preloaded from v08d19):
  - More decision patterns for the model to learn from
  - Better coverage of rare game states
  - No change to model architecture or objective

Risk: slightly longer training. Potential for mild overfitting if extra rows
are from the same episodes as validation. But validation is preloaded and fixed,
so the train/val split remains clean.

Based on v08d34 (truncation_level=1, position_winrate, top1=0.5485 BEST).

Changes from v08d34:
  1. EXPERIMENT_NAME → v0_08d40_more_data
  2. RUN_PREFIX → pokemon-20260627-v0-08d40
  3. Cell[7]: Preload from v08d19 (same split)
  4. Cell[19]: Increase max_rows cap from 350k to 420k
"""
import json

SRC = '/kaggle/working/pokemon-20260627-v0-08d34.ipynb'
DST = '/kaggle/working/pokemon-20260627-v0-08d40.ipynb'

with open(SRC) as f:
    nb = json.load(f)

cells = nb['cells']

# ── Cell[1]: Config ───────────────────────────────────────────────────────────
src1 = ''.join(cells[1]['source'])

OLD_EXP = "EXPERIMENT_NAME = 'v0_08d34_trunc1'"
NEW_EXP = "EXPERIMENT_NAME = 'v0_08d40_more_data'"
assert OLD_EXP in src1
src1 = src1.replace(OLD_EXP, NEW_EXP)

OLD_PFX = "RUN_PREFIX = 'pokemon-20260627-v0-08d34'"
NEW_PFX = "RUN_PREFIX = 'pokemon-20260627-v0-08d40'"
assert OLD_PFX in src1
src1 = src1.replace(OLD_PFX, NEW_PFX)

cells[1]['source'] = src1.splitlines(keepends=True)

# ── Cell[7]: Preload ──────────────────────────────────────────────────────────
src7 = ''.join(cells[7]['source'])

OLD_PRELOAD = "# v08d34: preload from v08d19 (same split)\n"
NEW_PRELOAD = "# v08d40: preload from v08d19 (same split)\n"
assert OLD_PRELOAD in src7
src7 = src7.replace(OLD_PRELOAD, NEW_PRELOAD)

OLD_PRINT = "    print(f'v08d34 preload: {len(DECISION_ROWS_DF)} decisions from {_pfx}')\n"
NEW_PRINT = "    print(f'v08d40 preload: {len(DECISION_ROWS_DF)} decisions from {_pfx}')\n"
assert OLD_PRINT in src7
src7 = src7.replace(OLD_PRINT, NEW_PRINT)

OLD_FAIL = "    print('v08d34 preload: no cache found, mining from scratch')\n"
NEW_FAIL = "    print('v08d40 preload: no cache found, mining from scratch')\n"
assert OLD_FAIL in src7
src7 = src7.replace(OLD_FAIL, NEW_FAIL)

cells[7]['source'] = src7.splitlines(keepends=True)

# ── Cell[19]: Increase max_rows cap 350k → 420k ────────────────────────────────
src19 = ''.join(cells[19]['source'])

OLD_MAXROWS = (
    "        max_rows = int(os.environ.get('V05_UNKNOWN0_MLP_MAX_ROWS',\n"
    "                       str(min(350_000, max(50_000, MLP_MAX_ROWS // 2)))))\n"
)
NEW_MAXROWS = (
    "        # v08d40: increased cap to 420k to use all ~411k UNKNOWN_0 rows\n"
    "        max_rows = int(os.environ.get('V05_UNKNOWN0_MLP_MAX_ROWS',\n"
    "                       str(min(420_000, max(50_000, MLP_MAX_ROWS)))))\n"
)
assert OLD_MAXROWS in src19, 'max_rows formula not found in Cell[19]'
src19 = src19.replace(OLD_MAXROWS, NEW_MAXROWS)

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

assert 'v0_08d40_more_data' in s1
assert "RUN_PREFIX = 'pokemon-20260627-v0-08d40'" in s1
assert 'v08d19' in s7
assert '420_000' in s19
assert '350_000' not in s19, 'old 350k cap should be replaced'
assert 'lambdarank_truncation_level' in s19
assert "lambdarank_truncation_level': 1" in s19
assert 'position_winrate' in s19
assert '_winner_weight = 4.0' in s19
full = ''.join(''.join(c['source']) for c in nb2['cells'])
assert 'lambdarank' in full
print('All sanity checks passed.')
