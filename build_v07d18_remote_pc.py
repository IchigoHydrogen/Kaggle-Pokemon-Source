"""Build pokemon-20260625-v0-07d18-remote-pc.ipynb from v07d17-remote-pc notebook.

v07d18: aggressive — compound RL round 3, warm start from v07d17 best model (iter110, +0.0490).

Changes from v07d17:
  [COMPOUND LEARNING ROUND 3]
    1. Warm start: replace v07d16 warm start with v07d17 iter110 model weights.
       v07d17 model is 97-dim (dim96 zero-padded); truncate to 96-dim.
       IL baseline should start at ~+0.0490 (vs v07d17's +0.0308 start).

  Everything else identical to v07d17 (which inherited from v07d16):
    - Quality-weighted IL sampling (rank_bucket × won)
    - N_ITERS=1000 + patience=150 early stopping
    - Memory safety stop at 58 GB
    - gc.collect removed
    - Episode data from v06d18
"""
import json

SRC = '/kaggle/working/pokemon-20260625-v0-07d17-remote-pc.ipynb'
DST = '/kaggle/working/pokemon-20260625-v0-07d18-remote-pc.ipynb'

with open(SRC) as f:
    nb = json.load(f)

cells = nb['cells']

# ── Cell[1]: Config ───────────────────────────────────────────────────────────
src1 = ''.join(cells[1]['source'])
src1 = src1.replace(
    "EXPERIMENT_NAME   = 'v0_07d17_compound_warmstart'",
    "EXPERIMENT_NAME   = 'v0_07d18_compound_warmstart_r3'"
)
src1 = src1.replace(
    "RUN_PREFIX = os.environ.get('V07D17_RUN_PREFIX', 'pokemon-20260625-v0-07d17-remote-pc')",
    "RUN_PREFIX = os.environ.get('V07D18_RUN_PREFIX', 'pokemon-20260625-v0-07d18-remote-pc')"
)
assert 'v0_07d18_compound_warmstart_r3' in src1, 'EXPERIMENT_NAME'
assert 'v0-07d18-remote-pc' in src1, 'RUN_PREFIX'
cells[1]['source'] = src1.splitlines(keepends=True)

# ── Cell[23]: RL training ─────────────────────────────────────────────────────
src23 = ''.join(cells[23]['source'])

# ── Patch 1: Update warm start path from v07d16 → v07d17 ─────────────────────
OLD_WS = \
    "    _v07d17_ws_path = Path('/kaggle/working/pokemon-20260625-v0-07d16-remote-pc/pokemon-20260625-v0-07d16-remote-pc-main_option_scorer.pt')\n" \
    "    _il_state2_97 = torch.load(str(_v07d17_ws_path), map_location='cpu', weights_only=True)\n" \
    "    _il_state2 = {k: (v[:, :96] if k == 'net.0.weight' else v) for k, v in _il_state2_97.items()}\n" \
    "    print(f'v07d17: warm start from v07d16 iter110 (+0.0308); net.0.weight: {_il_state2[\"net.0.weight\"].shape}')"
NEW_WS = \
    "    _v07d18_ws_path = Path('/kaggle/working/pokemon-20260625-v0-07d17-remote-pc/pokemon-20260625-v0-07d17-remote-pc-main_option_scorer.pt')\n" \
    "    _il_state2_97 = torch.load(str(_v07d18_ws_path), map_location='cpu', weights_only=True)\n" \
    "    _il_state2 = {k: (v[:, :96] if k == 'net.0.weight' else v) for k, v in _il_state2_97.items()}\n" \
    "    print(f'v07d18: warm start from v07d17 iter110 (+0.0490); net.0.weight: {_il_state2[\"net.0.weight\"].shape}')"
assert OLD_WS in src23, 'v07d17 warm start block not found'
src23 = src23.replace(OLD_WS, NEW_WS)

# ── Patch 2: Summary label ────────────────────────────────────────────────────
assert "print(f'\\n=== v0-07d17 Summary ===')" in src23, 'summary label'
src23 = src23.replace(
    "    print(f'\\n=== v0-07d17 Summary ===')",
    "    print(f'\\n=== v0-07d18 Summary ===')"
)

cells[23]['source'] = src23.splitlines(keepends=True)

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
s23 = ''.join(nb2['cells'][23]['source'])

assert 'v0_07d18_compound_warmstart_r3' in s1,     'EXPERIMENT_NAME'
assert 'v0-07d18-remote-pc' in s1,                  'RUN_PREFIX'
assert '_V07D2_FEAT_DIM      = 96' in s23,           'FEAT_DIM=96'
assert "'1000'))  # v07d16" in s23,                  'N_ITERS=1000'
assert '_il_weights_v7d2' in s23,                    'IL weights'
assert 'p=_il_probs_v7d2' in s23,                    'weighted IL sampling'
assert 'Memory safety stop' in s23,                  'memory safety stop'
assert "__import__('gc').collect()" not in s23,      'gc.collect removed'
assert '_v07d18_ws_path' in s23,                     'v07d18 warm start path'
assert 'v0-07d17-remote-pc' in s23,                  'v07d17 model path'
assert "v[:, :96]" in s23,                           '97->96 truncation'
assert 'v07d17 iter110' in s23,                      'warm start comment'
assert 'v0-07d18 Summary' in s23,                    'summary label'
assert '_v07d17_ws_path' not in s23,                 'no old v07d17 var name'
print('All sanity checks passed.')
