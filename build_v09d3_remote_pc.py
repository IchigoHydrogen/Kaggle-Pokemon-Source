"""Build pokemon-20260627-v0-09d3-remote-pc.ipynb from v09d2.

v09d3: also neutralize position_winrate skew (Track A, complete the skew kill).

v09d2 fixed op_last_context skew (train-on-proxy) → infer_top1 0.5098→0.5246.
But position_winrate STILL has train/serve skew: training uses the REAL computed
value, while inference hardcodes 0.5 (constant). So the model learns to rely on a
signal it never receives at inference.

FIX: set position_winrate = 0.5 in the TRAINING frame too (matching inference).
This makes it a constant → the model stops splitting on it → no skew.

Expectation: infer_top1 >= v09d2's 0.5246 (removing a misleading signal the model
can't use at inference should help or be neutral). Measured directly by the
infer-faithful eval (real_top1 should ≈ infer_top1 now that both major skews are gone).

Based on v09d2 (train-on-proxy op_last_context, infer_top1=0.5246).

Changes from v09d2:
  1. EXPERIMENT_NAME → v0_09d3_pwr_neutralize
  2. RUN_PREFIX → pokemon-20260627-v0-09d3
  3. Cell[7]: Preload from v08d19 (same split)
  4. Cell[19]: set work['position_winrate']=0.5 after its computation
"""
import json

SRC = '/kaggle/working/pokemon-20260627-v0-09d2.ipynb'
DST = '/kaggle/working/pokemon-20260627-v0-09d3.ipynb'

with open(SRC) as f:
    nb = json.load(f)

cells = nb['cells']

# ── Cell[1] ───────────────────────────────────────────────────────────────────
src1 = ''.join(cells[1]['source'])
src1 = src1.replace("EXPERIMENT_NAME = 'v0_09d2_train_proxy'",
                    "EXPERIMENT_NAME = 'v0_09d3_pwr_neutralize'")
src1 = src1.replace("RUN_PREFIX = 'pokemon-20260627-v0-09d2'",
                    "RUN_PREFIX = 'pokemon-20260627-v0-09d3'")
assert 'v0_09d3_pwr_neutralize' in src1
cells[1]['source'] = src1.splitlines(keepends=True)

# ── Cell[7] ───────────────────────────────────────────────────────────────────
src7 = ''.join(cells[7]['source'])
src7 = src7.replace("# v09d2: preload from v08d19 (same split)\n",
                    "# v09d3: preload from v08d19 (same split)\n")
src7 = src7.replace("    print(f'v09d2 preload: {len(DECISION_ROWS_DF)} decisions from {_pfx}')\n",
                    "    print(f'v09d3 preload: {len(DECISION_ROWS_DF)} decisions from {_pfx}')\n")
src7 = src7.replace("    print('v09d2 preload: no cache found, mining from scratch')\n",
                    "    print('v09d3 preload: no cache found, mining from scratch')\n")
cells[7]['source'] = src7.splitlines(keepends=True)

# ── Cell[19]: neutralize position_winrate in training ─────────────────────────
src19 = ''.join(cells[19]['source'])

ANCHOR = (
    "        work['position_winrate'] = work['position_winrate'].fillna(0.5)\n"
    "        work = work.drop(columns=['_turn_bucket'], errors='ignore')\n"
)
INSERT = (
    "        work['position_winrate'] = work['position_winrate'].fillna(0.5)\n"
    "        # v09d3: neutralize position_winrate skew — inference hardcodes 0.5,\n"
    "        # so training on the real value is pure train/serve skew. Make it constant.\n"
    "        work['position_winrate'] = 0.5\n"
    "        work = work.drop(columns=['_turn_bucket'], errors='ignore')\n"
)
assert ANCHOR in src19, 'position_winrate fillna anchor not found'
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
assert 'v0_09d3_pwr_neutralize' in s1
assert "RUN_PREFIX = 'pokemon-20260627-v0-09d3'" in s1
assert 'v08d19' in s7
assert "v09d3: neutralize position_winrate" in s19
assert "work['position_winrate'] = 0.5" in s19
assert 'train-on-proxy' in s19, 'v09d2 op_last_context proxy must be kept'
assert 'INFER-FAITHFUL' in s19, 'infer-faithful eval must be kept'
assert "lambdarank_truncation_level': 1" in s19
print('All sanity checks passed.')
