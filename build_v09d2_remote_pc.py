"""Build pokemon-20260627-v0-09d2-remote-pc.ipynb from v09d1.

v09d2: TRAIN-ON-PROXY for op_last_context (Track A — kill train/serve skew).

v09d1 revealed: real_top1=0.5485 but infer_top1=0.5098 (skew=+0.0387).
Cause: op_last_context proxy (from op_active_energy_count) differs from the real
mined value 94.5% of the time. The model was trained on the REAL op_last_context
(values like TO_BENCH dominate) but at inference is fed a 3-value op_energy proxy
that can NEVER output TO_BENCH. So the model's most important sequential feature
is garbage at inference.

FIX (this experiment): train the model on the SAME proxy the inference uses.
Override op_last_context in the training frame `work` with the op_energy-derived
proxy, and steps_since_op=2.0 (the inference constant). Now train == serve by
construction → skew for this feature → ~0.

Expectation:
  - real_top1 will DROP from 0.5485 (proxy is weaker than real op_last_context)
  - BUT infer_top1 should RISE above 0.5098 (model now uses a feature it actually
    receives correctly, instead of being fed unexpected values)
  - real_top1 ≈ infer_top1 (skew killed)
  The infer-faithful eval (added in v09d1) measures this directly.

This is the locally-measurable Track A win: improve infer_top1 without spending
a Kaggle submission. v09d3 will attempt the higher-ceiling fix (reconstruct the
REAL op_last_context from obs.logs at inference).

Based on v09d1 (v08d34 weights + infer-faithful eval).

Changes from v09d1:
  1. EXPERIMENT_NAME → v0_09d2_train_proxy
  2. RUN_PREFIX → pokemon-20260627-v0-09d2
  3. Cell[7]: Preload from v08d19 (same split)
  4. Cell[19]: override op_last_context/steps_since_op in `work` with inference proxy
"""
import json

SRC = '/kaggle/working/pokemon-20260627-v0-09d1.ipynb'
DST = '/kaggle/working/pokemon-20260627-v0-09d2.ipynb'

with open(SRC) as f:
    nb = json.load(f)

cells = nb['cells']

# ── Cell[1]: Config ───────────────────────────────────────────────────────────
src1 = ''.join(cells[1]['source'])

OLD_EXP = "EXPERIMENT_NAME = 'v0_09d1_infer_faithful'"
NEW_EXP = "EXPERIMENT_NAME = 'v0_09d2_train_proxy'"
assert OLD_EXP in src1
src1 = src1.replace(OLD_EXP, NEW_EXP)

OLD_PFX = "RUN_PREFIX = 'pokemon-20260627-v0-09d1'"
NEW_PFX = "RUN_PREFIX = 'pokemon-20260627-v0-09d2'"
assert OLD_PFX in src1
src1 = src1.replace(OLD_PFX, NEW_PFX)

cells[1]['source'] = src1.splitlines(keepends=True)

# ── Cell[7]: Preload ──────────────────────────────────────────────────────────
src7 = ''.join(cells[7]['source'])
src7 = src7.replace("# v09d1: preload from v08d19 (same split)\n",
                    "# v09d2: preload from v08d19 (same split)\n")
src7 = src7.replace("    print(f'v09d1 preload: {len(DECISION_ROWS_DF)} decisions from {_pfx}')\n",
                    "    print(f'v09d2 preload: {len(DECISION_ROWS_DF)} decisions from {_pfx}')\n")
src7 = src7.replace("    print('v09d1 preload: no cache found, mining from scratch')\n",
                    "    print('v09d2 preload: no cache found, mining from scratch')\n")
cells[7]['source'] = src7.splitlines(keepends=True)

# ── Cell[19]: override op_last_context with inference proxy ────────────────────
src19 = ''.join(cells[19]['source'])

ANCHOR = (
    "        else:\n"
    "            work['op_last_context'] = 'NONE'\n"
    "            work['steps_since_op'] = -1.0\n"
    "        # v08d28: position_winrate"
)
INSERT = (
    "        else:\n"
    "            work['op_last_context'] = 'NONE'\n"
    "            work['steps_since_op'] = -1.0\n"
    "        # v09d2: TRAIN-ON-PROXY — override op_last_context/steps_since_op with the\n"
    "        # EXACT proxy main.py uses at inference (from op_active_energy_count),\n"
    "        # killing the train/serve skew measured in v09d1 (skew=+0.0387).\n"
    "        if 'op_active_energy_count' in work.columns:\n"
    "            _oae_tr = work['op_active_energy_count'].fillna(0).astype(float)\n"
    "            work['op_last_context'] = _oae_tr.map(\n"
    "                lambda e: 'UNKNOWN_0' if e >= 2 else ('ATTACH_FROM' if e == 1 else 'NONE'))\n"
    "            work['steps_since_op'] = 2.0\n"
    "            print(f'v09d2 train-on-proxy op_last_context dist: '\n"
    "                  f'{work[\"op_last_context\"].value_counts().to_dict()}')\n"
    "        else:\n"
    "            print('v09d2 train-on-proxy: op_active_energy_count missing, no override')\n"
    "        # v08d28: position_winrate"
)
assert ANCHOR in src19, 'op_last_context anchor not found'
src19 = src19.replace(ANCHOR, INSERT)

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

assert 'v0_09d2_train_proxy' in s1
assert "RUN_PREFIX = 'pokemon-20260627-v0-09d2'" in s1
assert 'v08d19' in s7
assert 'train-on-proxy' in s19
assert 'INFER-FAITHFUL' in s19, 'infer-faithful eval from v09d1 must be kept'
assert "lambdarank_truncation_level': 1" in s19
assert 'position_winrate' in s19
assert '_winner_weight = 4.0' in s19
print('All sanity checks passed.')
