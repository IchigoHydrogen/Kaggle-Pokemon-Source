"""Build pokemon-20260627-v0-08d25-remote-pc.ipynb from v08d19.

v08d25: is_post_ko + prize_urgency features (error-analysis-driven).

ERROR ANALYSIS FINDINGS from v08d19 validation:
  - op_last_context='TO_BENCH' (post-KO): 53% of decisions, top1=0.524 (weakest)
  - op_last_context='UNKNOWN_0' (mid-game): 41% of decisions, top1=0.573
  - prize_gap=5 (winning by a lot): top1=0.404 (catastrophically bad)
  - prize_gap=3: top1=0.478 (worse than balanced game)
  Balanced game (prize_gap=0) is BEST: top1=0.580

Two new features addressing identified failure modes:
  1. is_post_ko: binary 1 if op_last_context in {TO_BENCH, ATTACH_FROM}
     → explicitly marks post-KO recovery situations for specialized split
     → inference proxy: 1 if op_energy==1 (ATTACH_FROM via energy count)
  2. prize_urgency = abs(prize_gap)
     → captures game imbalance regardless of direction
     → fully inference-safe (computed directly from observable prize counts)

Changes from v08d19:
  1. EXPERIMENT_NAME → v0_08d25_postkobf
  2. RUN_PREFIX → pokemon-20260627-v0-08d25
  3. Cell[7]: Preload from v08d19 (same split)
  4. Cell[19]: Add is_post_ko + prize_urgency to work + extra_numeric_candidates
  5. Cell[19]: Add inference proxy for both features
"""
import json

SRC = '/kaggle/working/pokemon-20260627-v0-08d19.ipynb'
DST = '/kaggle/working/pokemon-20260627-v0-08d25.ipynb'

with open(SRC) as f:
    nb = json.load(f)

cells = nb['cells']

# ── Cell[1]: Config ───────────────────────────────────────────────────────────
src1 = ''.join(cells[1]['source'])

OLD_EXP = "EXPERIMENT_NAME = 'v0_08d19_op_seq'"
NEW_EXP = "EXPERIMENT_NAME = 'v0_08d25_postkobf'"
assert OLD_EXP in src1
src1 = src1.replace(OLD_EXP, NEW_EXP)

OLD_PFX = "RUN_PREFIX = 'pokemon-20260627-v0-08d19'"
NEW_PFX = "RUN_PREFIX = 'pokemon-20260627-v0-08d25'"
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
    "# v08d25: preload from v08d19 (same split)\n"
    "_PRELOAD_CANDIDATES = [\n"
    "    Path('/kaggle/working/pokemon-20260627-v0-08d19/pokemon-20260627-v0-08d19'),\n"
    "    Path('/kaggle/working/pokemon-20260627-v0-08d18/pokemon-20260627-v0-08d18'),\n"
    "]\n"
)
assert OLD_PRELOAD in src7
src7 = src7.replace(OLD_PRELOAD, NEW_PRELOAD)

OLD_PRINT = "    print(f'v08d19 preload: {len(DECISION_ROWS_DF)} decisions from {_pfx}')\n"
NEW_PRINT = "    print(f'v08d25 preload: {len(DECISION_ROWS_DF)} decisions from {_pfx}')\n"
assert OLD_PRINT in src7
src7 = src7.replace(OLD_PRINT, NEW_PRINT)

OLD_FAIL = "    print('v08d19 preload: v08d18 cache not found, mining from scratch')\n"
NEW_FAIL = "    print('v08d25 preload: no cache found, mining from scratch')\n"
assert OLD_FAIL in src7
src7 = src7.replace(OLD_FAIL, NEW_FAIL)

cells[7]['source'] = src7.splitlines(keepends=True)

# ── Cell[19]: Add is_post_ko + prize_urgency ──────────────────────────────────
src19 = ''.join(cells[19]['source'])

# 1. Add feature computation after op_last_context is ready, before extra_numeric
OLD_EXTRA_BLOCK = (
    "        else:\n"
    "            work['op_last_context'] = 'NONE'\n"
    "            work['steps_since_op'] = -1.0\n"
    "        # Extra features from ALAKAZAM_OPTION_MODEL_DF\n"
    "        _extra_numeric_candidates = ['supporter_played', 'stadium_played', 'energy_attached',\n"
    "                                      'my_discard_count', 'op_discard_count', 'prize_gap', 'steps_since_op']\n"
)
NEW_EXTRA_BLOCK = (
    "        else:\n"
    "            work['op_last_context'] = 'NONE'\n"
    "            work['steps_since_op'] = -1.0\n"
    "        # v08d25: is_post_ko (post-KO recovery indicator) and prize_urgency\n"
    "        work['is_post_ko'] = work['op_last_context'].isin(['TO_BENCH', 'ATTACH_FROM']).astype(float)\n"
    "        work['prize_urgency'] = work['prize_gap'].abs() if 'prize_gap' in work.columns else 0.0\n"
    "        print(f'v08d25 is_post_ko: rate={work[\"is_post_ko\"].mean():.4f} ({int(work[\"is_post_ko\"].sum())} rows)')\n"
    "        print(f'v08d25 prize_urgency: mean={work[\"prize_urgency\"].mean():.2f}, max={work[\"prize_urgency\"].max():.0f}')\n"
    "        # Extra features from ALAKAZAM_OPTION_MODEL_DF\n"
    "        _extra_numeric_candidates = ['supporter_played', 'stadium_played', 'energy_attached',\n"
    "                                      'my_discard_count', 'op_discard_count', 'prize_gap', 'steps_since_op',\n"
    "                                      'is_post_ko', 'prize_urgency']\n"
)
assert OLD_EXTRA_BLOCK in src19, 'extra_block not found in Cell[19]'
src19 = src19.replace(OLD_EXTRA_BLOCK, NEW_EXTRA_BLOCK)

# 2. Inference proxy: add is_post_ko and prize_urgency after _prize_gap computation
# The inference row dict is escaped inside a string literal in the cell source
OLD_INFER_DICT = (
    "\\'prize_gap\\': _prize_gap,\\n"
    "                \\'op_last_context\\': _op_last_ctx,\\n"
    "                \\'steps_since_op\\': _steps_since_op,\\n"
)
NEW_INFER_DICT = (
    "\\'prize_gap\\': _prize_gap,\\n"
    "                \\'op_last_context\\': _op_last_ctx,\\n"
    "                \\'steps_since_op\\': _steps_since_op,\\n"
    "                \\'is_post_ko\\': 1.0 if _op_energy == 1 else 0.0,\\n"
    "                \\'prize_urgency\\': abs(_prize_gap),\\n"
)
assert OLD_INFER_DICT in src19, f'inference dict not found in Cell[19]'
src19 = src19.replace(OLD_INFER_DICT, NEW_INFER_DICT)

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

assert 'v0_08d25_postkobf' in s1
assert "RUN_PREFIX = 'pokemon-20260627-v0-08d25'" in s1
assert 'v08d19' in s7
assert 'is_post_ko' in s19, 'is_post_ko not in Cell[19]'
assert 'prize_urgency' in s19, 'prize_urgency not in Cell[19]'
assert "'is_post_ko', 'prize_urgency'" in s19, 'features not in extra_numeric_candidates'
assert "\\'is_post_ko\\':" in s19 and "abs(_prize_gap)" in s19, 'inference proxy not found'
assert 'op_last_context' in s19, 'op_last_context kept'
assert 'prize_gap' in s19, 'prize_gap kept'
assert '_winner_weight = 4.0' in s19, 'winner_weight=4x kept'
full = ''.join(''.join(c['source']) for c in nb2['cells'])
assert 'lambdarank' in full
print('All sanity checks passed.')
