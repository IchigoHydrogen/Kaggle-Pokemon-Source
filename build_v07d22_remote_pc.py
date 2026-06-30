"""Build pokemon-20260625-v0-07d22-remote-pc.ipynb from v07d19-remote-pc notebook.

v07d22: aggressive — Pure IL mode + lower LR (3e-5, was 3e-4) + warm start v07d21 (+0.0602).

v07d21 key finding:
  Pure IL mode (LAMBDA_RL=0) broke the +0.049 ceiling → +0.0602 at best_iter=100!
  Net gain +0.0108 over v07d19 warm start (+0.0494).
  After iter=100: oscillated 0.001-0.037 → LR 3e-4 too large for fine-tuning.
  → Lower LR to allow finer convergence near the +0.06 optimum.

Changes from v07d19:
  1. LAMBDA_RL = 0.0 (keep pure IL mode — confirmed beneficial)
  2. GAMES_PER_ITER = 1 (keep pure IL overhead)
  3. LR: 3e-4 → 3e-5 (10x reduction for finer convergence)
  4. PATIENCE: 150 → 200 (more room at lower LR)
  5. Warm start: v07d17 → v07d21 best (iter100, +0.0602)
  LambdaRank (λ=0.5) retained.
"""
import json

SRC = '/kaggle/working/pokemon-20260625-v0-07d19-remote-pc.ipynb'
DST = '/kaggle/working/pokemon-20260625-v0-07d22-remote-pc.ipynb'

with open(SRC) as f:
    nb = json.load(f)

cells = nb['cells']

# ── Cell[1]: Config ───────────────────────────────────────────────────────────
src1 = ''.join(cells[1]['source'])
src1 = src1.replace(
    "EXPERIMENT_NAME   = 'v0_07d19_lambdarank_il'",
    "EXPERIMENT_NAME   = 'v0_07d22_pure_il_lowlr'"
)
src1 = src1.replace(
    "RUN_PREFIX = os.environ.get('V07D19_RUN_PREFIX', 'pokemon-20260625-v0-07d19-remote-pc')",
    "RUN_PREFIX = os.environ.get('V07D22_RUN_PREFIX', 'pokemon-20260625-v0-07d22-remote-pc')"
)
assert 'v0_07d22_pure_il_lowlr' in src1, 'EXPERIMENT_NAME'
assert 'v0-07d22-remote-pc' in src1, 'RUN_PREFIX'
cells[1]['source'] = src1.splitlines(keepends=True)

# ── Cell[23]: RL training ─────────────────────────────────────────────────────
src23 = ''.join(cells[23]['source'])

# ── Patch 1: Warm start path v07d17 → v07d21 ─────────────────────────────────
OLD_WS = \
    "    _v07d19_ws_path = Path('/kaggle/working/pokemon-20260625-v0-07d17-remote-pc/pokemon-20260625-v0-07d17-remote-pc-main_option_scorer.pt')\n" \
    "    _il_state2_97 = torch.load(str(_v07d19_ws_path), map_location='cpu', weights_only=True)\n" \
    "    _il_state2 = {k: (v[:, :96] if k == 'net.0.weight' else v) for k, v in _il_state2_97.items()}\n" \
    "    print(f'v07d19: warm start from v07d17 best model iter110 (+0.0490); net.0.weight: {_il_state2[\"net.0.weight\"].shape}')"
NEW_WS = \
    "    _v07d22_ws_path = Path('/kaggle/working/pokemon-20260625-v0-07d21-remote-pc/pokemon-20260625-v0-07d21-remote-pc-main_option_scorer.pt')\n" \
    "    _il_state2_97 = torch.load(str(_v07d22_ws_path), map_location='cpu', weights_only=True)\n" \
    "    _il_state2 = {k: (v[:, :96] if k == 'net.0.weight' else v) for k, v in _il_state2_97.items()}\n" \
    "    print(f'v07d22: warm start from v07d21 best model iter100 (+0.0602); net.0.weight: {_il_state2[\"net.0.weight\"].shape}')"
assert OLD_WS in src23, 'warm start block not found'
src23 = src23.replace(OLD_WS, NEW_WS)

# ── Patch 2: LAMBDA_RL = 0.0 ─────────────────────────────────────────────────
OLD_LAMBDA_RL = "    _V07D2_LAMBDA_RL    = 0.05"
NEW_LAMBDA_RL = "    _V07D2_LAMBDA_RL    = 0.0   # v07d22: pure IL mode (confirmed better)"
assert OLD_LAMBDA_RL in src23, 'LAMBDA_RL line not found'
src23 = src23.replace(OLD_LAMBDA_RL, NEW_LAMBDA_RL)

# ── Patch 3: GAMES_PER_ITER = 1 ──────────────────────────────────────────────
OLD_GAMES = \
    "            _V07D2_GAMES_PER_ITER = int(os.environ.get('V07D2_GAMES_PER_ITER', '200'))"
NEW_GAMES = \
    "            _V07D2_GAMES_PER_ITER = int(os.environ.get('V07D2_GAMES_PER_ITER', '1'))   # v07d22: pure IL"
assert OLD_GAMES in src23, 'GAMES_PER_ITER line not found'
src23 = src23.replace(OLD_GAMES, NEW_GAMES)

# ── Patch 4: LR 3e-4 → 3e-5 ─────────────────────────────────────────────────
OLD_LR = "    _V07D2_LR           = 3e-4"
NEW_LR = "    _V07D2_LR           = 3e-5  # v07d22: 10x lower for finer convergence near +0.06"
assert OLD_LR in src23, 'LR line not found'
src23 = src23.replace(OLD_LR, NEW_LR)

# ── Patch 5: PATIENCE 150 → 200 ──────────────────────────────────────────────
OLD_PATIENCE = "            _V07D2_PATIENCE       = int(os.environ.get('V07D2_PATIENCE',       '150'))   # v07d16: early stopping patience"
NEW_PATIENCE = "            _V07D2_PATIENCE       = int(os.environ.get('V07D2_PATIENCE',       '200'))   # v07d22: more room at lower LR"
assert OLD_PATIENCE in src23, 'PATIENCE line not found'
src23 = src23.replace(OLD_PATIENCE, NEW_PATIENCE)

# ── Patch 6: Summary label ────────────────────────────────────────────────────
assert "print(f'\\n=== v0-07d19 Summary ===')" in src23, 'summary label'
src23 = src23.replace(
    "    print(f'\\n=== v0-07d19 Summary ===')",
    "    print(f'\\n=== v0-07d22 Summary ===')"
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

assert 'v0_07d22_pure_il_lowlr' in s1,            'EXPERIMENT_NAME'
assert 'v0-07d22-remote-pc' in s1,                 'RUN_PREFIX'
assert '_V07D2_FEAT_DIM      = 96' in s23,          'FEAT_DIM=96'
assert "'1000'))  # v07d16" in s23,                 'N_ITERS=1000'
assert '_V07D2_LAMBDA_RL    = 0.0' in s23,          'LAMBDA_RL=0'
assert "'1'))   # v07d22: pure IL" in s23,          'GAMES_PER_ITER=1'
assert '_V07D2_LR           = 3e-5' in s23,         'LR=3e-5'
assert "'200'))   # v07d22" in s23,                  'PATIENCE=200'
assert '_V07D2_LAMBDA_RANK  = 0.5' in s23,          'LAMBDA_RANK=0.5'
assert '_il_won_ep' in s23,                         'LambdaRank won tracking'
assert 'F.logsigmoid' in s23,                       'LambdaRank loss'
assert '_v07d22_ws_path' in s23,                    'v07d22 warm start path'
assert 'v0-07d21-remote-pc' in s23,                 'v07d21 model path'
assert '_v07d19_ws_path' not in s23,                'no old var name'
assert '_v07d2_sp_buffer' not in s23,               'no SP buffer'
assert 'v0-07d22 Summary' in s23,                   'summary label'
assert 'Memory safety stop' in s23,                 'memory safety stop'
print('All sanity checks passed.')
