"""Build pokemon-20260627-v0-08d20-remote-pc.ipynb from v08d19.

v08d20: scale up after op_last_context breakthrough.

v08d19 found op_last_context is the 7th most important feature (+0.0229 top1 gain).
Now give the model more capacity and more data to exploit this:

  num_leaves: 63 → 127   (2x model capacity, capture interaction patterns better)
  max_rows:   350k → 700k (2x training data)
  steps_since_op: REMOVED (always 0 due to same-step decisions, no information)

Everything else from v08d19 is kept (op_last_context, prize_gap, opp_arch).

Changes from v08d19:
  1. EXPERIMENT_NAME → v0_08d20_scale_up
  2. RUN_PREFIX → pokemon-20260627-v0-08d20
  3. Cell[7]: Preload from v08d19
  4. Cell[19]: num_leaves 63 → 127
  5. Cell[19]: max_rows 350_000 → 700_000
  6. Cell[19]: remove steps_since_op from _extra_numeric_candidates
  7. Cell[19]: remove steps_since_op from injection code
"""
import json

SRC = '/kaggle/working/pokemon-20260627-v0-08d19.ipynb'
DST = '/kaggle/working/pokemon-20260627-v0-08d20.ipynb'

with open(SRC) as f:
    nb = json.load(f)

cells = nb['cells']

# ── Cell[1]: Config ───────────────────────────────────────────────────────────
src1 = ''.join(cells[1]['source'])

OLD_EXP = "EXPERIMENT_NAME = 'v0_08d19_op_seq'"
NEW_EXP = "EXPERIMENT_NAME = 'v0_08d20_scale_up'"
assert OLD_EXP in src1
src1 = src1.replace(OLD_EXP, NEW_EXP)

OLD_PFX = "RUN_PREFIX = 'pokemon-20260627-v0-08d19'"
NEW_PFX = "RUN_PREFIX = 'pokemon-20260627-v0-08d20'"
assert OLD_PFX in src1
src1 = src1.replace(OLD_PFX, NEW_PFX)

# max_rows: MAX_CONTEXT_TRAIN_ROWS 350_000 → 700_000
OLD_MAX = "    MAX_CONTEXT_TRAIN_ROWS = 350_000\n    MAX_GLOBAL_TRAIN_ROWS = 700_000"
NEW_MAX = "    MAX_CONTEXT_TRAIN_ROWS = 700_000\n    MAX_GLOBAL_TRAIN_ROWS = 1_400_000"
assert OLD_MAX in src1, 'MAX_CONTEXT_TRAIN_ROWS not found in Cell[1]'
src1 = src1.replace(OLD_MAX, NEW_MAX)

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
    "# v08d20: preload from v08d19\n"
    "_PRELOAD_CANDIDATES = [\n"
    "    Path('/kaggle/working/pokemon-20260627-v0-08d19/pokemon-20260627-v0-08d19'),\n"
    "    Path('/kaggle/working/pokemon-20260627-v0-08d18/pokemon-20260627-v0-08d18'),\n"
    "]\n"
)
assert OLD_PRELOAD in src7, 'preload candidates not found in Cell[7]'
src7 = src7.replace(OLD_PRELOAD, NEW_PRELOAD)

OLD_PRINT = "    print(f'v08d19 preload: {len(DECISION_ROWS_DF)} decisions from {_pfx}')\n"
NEW_PRINT = "    print(f'v08d20 preload: {len(DECISION_ROWS_DF)} decisions from {_pfx}')\n"
assert OLD_PRINT in src7
src7 = src7.replace(OLD_PRINT, NEW_PRINT)

OLD_FAIL = "    print('v08d19 preload: v08d18 cache not found, mining from scratch')\n"
NEW_FAIL = "    print('v08d20 preload: v08d19/v08d18 cache not found, mining from scratch')\n"
assert OLD_FAIL in src7
src7 = src7.replace(OLD_FAIL, NEW_FAIL)

cells[7]['source'] = src7.splitlines(keepends=True)

# ── Cell[19]: num_leaves + max_rows + remove steps_since_op ──────────────────
src19 = ''.join(cells[19]['source'])

# 1. num_leaves: 63 → 127
OLD_LEAVES = "'num_leaves': 63,"
NEW_LEAVES = "'num_leaves': 127,"
assert OLD_LEAVES in src19, 'num_leaves not found'
src19 = src19.replace(OLD_LEAVES, NEW_LEAVES)

# (max_rows is configured via MAX_CONTEXT_TRAIN_ROWS in Cell[1], patched below)

# 3. Remove steps_since_op from extra_numeric_candidates
OLD_EXTRAS = (
    "        _extra_numeric_candidates = ['supporter_played', 'stadium_played', 'energy_attached',\n"
    "                                      'my_discard_count', 'op_discard_count', 'prize_gap', 'steps_since_op']"
)
NEW_EXTRAS = (
    "        _extra_numeric_candidates = ['supporter_played', 'stadium_played', 'energy_attached',\n"
    "                                      'my_discard_count', 'op_discard_count', 'prize_gap']"
)
assert OLD_EXTRAS in src19, 'steps_since_op in extra_numeric not found'
src19 = src19.replace(OLD_EXTRAS, NEW_EXTRAS)

# 4. Remove steps_since_op from injection code
OLD_STEPS_INJ = (
    "        _steps_since_op = 2.0  # v08d19: approx (alternating turns)\\n"
    "        rows = []\\n        for i, o in enumerate(select.option):"
)
NEW_STEPS_INJ = (
    "        rows = []\\n        for i, o in enumerate(select.option):"
)
assert OLD_STEPS_INJ in src19, 'steps_since_op injection not found'
src19 = src19.replace(OLD_STEPS_INJ, NEW_STEPS_INJ)

# 5. Remove steps_since_op from row dict
OLD_STEPS_ROW = (
    "                \\'op_last_context\\': _op_last_ctx,\\n"
    "                \\'steps_since_op\\': _steps_since_op,\\n"
    "            })"
)
NEW_STEPS_ROW = (
    "                \\'op_last_context\\': _op_last_ctx,\\n"
    "            })"
)
assert OLD_STEPS_ROW in src19, 'steps_since_op row dict not found'
src19 = src19.replace(OLD_STEPS_ROW, NEW_STEPS_ROW)

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

assert 'v0_08d20_scale_up' in s1
assert "RUN_PREFIX = 'pokemon-20260627-v0-08d20'" in s1
assert 'v08d19' in s7,       'preload from v08d19'
assert "'num_leaves': 127" in s19, 'num_leaves=127'
assert 'MAX_CONTEXT_TRAIN_ROWS = 700_000' in s1, 'max_rows=700k'
assert "'prize_gap', 'steps_since_op'" not in s19, 'steps_since_op not removed from extra_numeric'
assert "_steps_since_op" not in s19, '_steps_since_op not removed from injection'
assert 'op_last_context' in s19, 'op_last_context kept'
assert 'prize_gap' in s19, 'prize_gap kept'
assert 'opponent_archetype_norm' in s19, 'opp_arch kept'
full = ''.join(''.join(c['source']) for c in nb2['cells'])
assert 'lambdarank' in full
print('All sanity checks passed.')
