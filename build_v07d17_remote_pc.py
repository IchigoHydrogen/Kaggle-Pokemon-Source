"""Build pokemon-20260625-v0-07d17-remote-pc.ipynb from v07d16-remote-pc notebook.

v07d17: aggressive — compound RL warm start from v07d16 best model (iter110, +0.0308).

Changes from v07d16:
  [COMPOUND LEARNING]
    1. Warm start: replace IL model init with v07d16 iter110 model weights.
       v07d16 model is 97-dim (dim96 zero-padded for Part B compat); truncate to 96-dim.
       Effect: RL starts at winner_margin≈+0.0308 instead of near 0.
       Episode data (feature matrix, decision rows) still from v06d18.

  Everything else identical to v07d16:
    - Quality-weighted IL sampling (rank_bucket × won)
    - N_ITERS=1000 + patience=150 early stopping
    - Memory safety stop at 58 GB
    - gc.collect removed
"""
import json

SRC = '/kaggle/working/pokemon-20260625-v0-07d16-remote-pc.ipynb'
DST = '/kaggle/working/pokemon-20260625-v0-07d17-remote-pc.ipynb'

with open(SRC) as f:
    nb = json.load(f)

cells = nb['cells']

# ── Cell[1]: Config ───────────────────────────────────────────────────────────
src1 = ''.join(cells[1]['source'])
src1 = src1.replace(
    "EXPERIMENT_NAME   = 'v0_07d16_il_weighted_1000iter'",
    "EXPERIMENT_NAME   = 'v0_07d17_compound_warmstart'"
)
src1 = src1.replace(
    "RUN_PREFIX = os.environ.get('V07D16_RUN_PREFIX', 'pokemon-20260625-v0-07d16-remote-pc')",
    "RUN_PREFIX = os.environ.get('V07D17_RUN_PREFIX', 'pokemon-20260625-v0-07d17-remote-pc')"
)
assert 'v0_07d17_compound_warmstart' in src1, 'EXPERIMENT_NAME'
assert 'v0-07d17-remote-pc' in src1, 'RUN_PREFIX'
cells[1]['source'] = src1.splitlines(keepends=True)

# ── Cell[23]: RL training ─────────────────────────────────────────────────────
src23 = ''.join(cells[23]['source'])

# ── Patch 1: Replace IL state2 load with v07d16 warm start ───────────────────
OLD_IL_LOAD = \
    "    _il_state2  = torch.load(str(_il_model_path2), map_location='cpu', weights_only=True)"
NEW_IL_LOAD = """\
    # v07d17: compound RL warm start — load v07d16 best model (iter110, +0.0308)
    # v07d16 model is 97-dim (dim96 zero-padded); truncate net.0.weight to 96-dim
    _v07d17_ws_path = Path('/kaggle/working/pokemon-20260625-v0-07d16-remote-pc/pokemon-20260625-v0-07d16-remote-pc-main_option_scorer.pt')
    _il_state2_97 = torch.load(str(_v07d17_ws_path), map_location='cpu', weights_only=True)
    _il_state2 = {k: (v[:, :96] if k == 'net.0.weight' else v) for k, v in _il_state2_97.items()}
    print(f'v07d17: warm start from v07d16 iter110 (+0.0308); net.0.weight: {_il_state2[\"net.0.weight\"].shape}')"""
assert OLD_IL_LOAD in src23, 'il_state2 torch.load line not found'
src23 = src23.replace(OLD_IL_LOAD, NEW_IL_LOAD)

# ── Patch 2: Summary label ────────────────────────────────────────────────────
assert "print(f'\\n=== v0-07d16 Summary ===')" in src23, 'summary label'
src23 = src23.replace(
    "    print(f'\\n=== v0-07d16 Summary ===')",
    "    print(f'\\n=== v0-07d17 Summary ===')"
)

cells[23]['source'] = src23.splitlines(keepends=True)

# ── Clear outputs (code cells only) ──────────────────────────────────────────
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

assert 'v0_07d17_compound_warmstart' in s1,            'EXPERIMENT_NAME'
assert 'v0-07d17-remote-pc' in s1,                     'RUN_PREFIX'
assert '_V07D2_FEAT_DIM      = 96' in s23,              'FEAT_DIM=96'
assert "'1000'))  # v07d16" in s23,                     'N_ITERS=1000'
assert '_V07D2_PATIENCE       = 9999' in s23,           'PATIENCE smoke mode'
assert '_il_weights_v7d2' in s23,                       'IL weights'
assert 'p=_il_probs_v7d2' in s23,                       'weighted IL sampling'
assert '_v07d2_no_improve  = 0' in s23,                 'no_improve counter'
assert 'Memory safety stop' in s23,                     'memory safety stop'
assert "__import__('gc').collect()" not in s23,         'gc.collect removed'
assert '_v07d17_ws_path' in s23,                        'v07d17 warm start path'
assert "v[:, :96]" in s23,                              '97->96 truncation'
assert 'net.0.weight' in s23,                           'weight key'
assert 'v07d16 iter110' in s23,                         'warm start comment'
assert 'v07d11: re-saved as 97-dim' in s23,             '97-dim save logic'
assert 'v0-07d17 Summary' in s23,                       'summary label'
print('All sanity checks passed.')
