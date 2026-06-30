"""Build pokemon-20260625-v0-07d21-remote-pc.ipynb from v07d19-remote-pc notebook.

v07d21: aggressive — Pure IL mode (LAMBDA_RL=0).

Changes from v07d19:
  [BIG WEAPON: Remove RL interference]
    1. LAMBDA_RL = 0.0 (was 0.05) — disable RL gradient entirely.
    2. GAMES_PER_ITER = 1 (was 200) — collect 1 game/iter for loop compat.
       With 1 game, RL loss = 0 * (tiny PPO loss) = 0. Only IL trains.
    3. Warm start from v07d19 (+0.0494, best so far; better than v07d20 +0.0465).

  LambdaRank (v07d19) retained, SP buffer (v07d20) NOT included.

Rationale:
  RL gain per round: v07d16 +0.0308, v07d17 +0.0182, v07d18 -0.0006,
  v07d19 +0.0004, v07d20 SP -0.003.
  RL is at best neutral, at worst actively hurting holdout winner_margin.
  The RL trains on online game experience (different distribution from v06d18 holdout).
  IL directly optimizes the holdout distribution.
  Pure IL for 1000 iters (patience=150 early stop) tests whether the ceiling
  is the RL interference or the data ceiling.

  With GAMES_PER_ITER=1: ~3 sec/iter → 1000 iters = ~50 min.
  Memory barely grows (1 game deleted per iter).
  Total run: ~25 min pipeline + ~50 min IL = ~75 min.
"""
import json

SRC = '/kaggle/working/pokemon-20260625-v0-07d19-remote-pc.ipynb'
DST = '/kaggle/working/pokemon-20260625-v0-07d21-remote-pc.ipynb'

with open(SRC) as f:
    nb = json.load(f)

cells = nb['cells']

# ── Cell[1]: Config ───────────────────────────────────────────────────────────
src1 = ''.join(cells[1]['source'])
src1 = src1.replace(
    "EXPERIMENT_NAME   = 'v0_07d19_lambdarank_il'",
    "EXPERIMENT_NAME   = 'v0_07d21_pure_il_no_rl'"
)
src1 = src1.replace(
    "RUN_PREFIX = os.environ.get('V07D19_RUN_PREFIX', 'pokemon-20260625-v0-07d19-remote-pc')",
    "RUN_PREFIX = os.environ.get('V07D21_RUN_PREFIX', 'pokemon-20260625-v0-07d21-remote-pc')"
)
assert 'v0_07d21_pure_il_no_rl' in src1, 'EXPERIMENT_NAME'
assert 'v0-07d21-remote-pc' in src1, 'RUN_PREFIX'
cells[1]['source'] = src1.splitlines(keepends=True)

# ── Cell[23]: RL training ─────────────────────────────────────────────────────
src23 = ''.join(cells[23]['source'])

# ── Patch 1: Warm start path v07d17 → v07d19 ─────────────────────────────────
OLD_WS = \
    "    _v07d19_ws_path = Path('/kaggle/working/pokemon-20260625-v0-07d17-remote-pc/pokemon-20260625-v0-07d17-remote-pc-main_option_scorer.pt')\n" \
    "    _il_state2_97 = torch.load(str(_v07d19_ws_path), map_location='cpu', weights_only=True)\n" \
    "    _il_state2 = {k: (v[:, :96] if k == 'net.0.weight' else v) for k, v in _il_state2_97.items()}\n" \
    "    print(f'v07d19: warm start from v07d17 best model iter110 (+0.0490); net.0.weight: {_il_state2[\"net.0.weight\"].shape}')"
NEW_WS = \
    "    _v07d21_ws_path = Path('/kaggle/working/pokemon-20260625-v0-07d19-remote-pc/pokemon-20260625-v0-07d19-remote-pc-main_option_scorer.pt')\n" \
    "    _il_state2_97 = torch.load(str(_v07d21_ws_path), map_location='cpu', weights_only=True)\n" \
    "    _il_state2 = {k: (v[:, :96] if k == 'net.0.weight' else v) for k, v in _il_state2_97.items()}\n" \
    "    print(f'v07d21: warm start from v07d19 best model (+0.0494); net.0.weight: {_il_state2[\"net.0.weight\"].shape}')"
assert OLD_WS in src23, 'warm start block not found'
src23 = src23.replace(OLD_WS, NEW_WS)

# ── Patch 2: LAMBDA_RL = 0.0 (disable RL gradient) ───────────────────────────
OLD_LAMBDA_RL = \
    "    _V07D2_LAMBDA_RL    = 0.05"
NEW_LAMBDA_RL = \
    "    _V07D2_LAMBDA_RL    = 0.0   # v07d21: pure IL mode — no RL gradient"
assert OLD_LAMBDA_RL in src23, 'LAMBDA_RL line not found'
src23 = src23.replace(OLD_LAMBDA_RL, NEW_LAMBDA_RL)

# ── Patch 3: GAMES_PER_ITER = 1 (minimal game collection) ────────────────────
OLD_GAMES = \
    "            _V07D2_GAMES_PER_ITER = int(os.environ.get('V07D2_GAMES_PER_ITER', '200'))"
NEW_GAMES = \
    "            _V07D2_GAMES_PER_ITER = int(os.environ.get('V07D2_GAMES_PER_ITER', '1'))   # v07d21: pure IL"
assert OLD_GAMES in src23, 'GAMES_PER_ITER line not found'
src23 = src23.replace(OLD_GAMES, NEW_GAMES)

# ── Patch 4: Summary label ────────────────────────────────────────────────────
assert "print(f'\\n=== v0-07d19 Summary ===')" in src23, 'summary label'
src23 = src23.replace(
    "    print(f'\\n=== v0-07d19 Summary ===')",
    "    print(f'\\n=== v0-07d21 Summary ===')"
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

assert 'v0_07d21_pure_il_no_rl' in s1,           'EXPERIMENT_NAME'
assert 'v0-07d21-remote-pc' in s1,                'RUN_PREFIX'
assert '_V07D2_FEAT_DIM      = 96' in s23,         'FEAT_DIM=96'
assert "'1000'))  # v07d16" in s23,                'N_ITERS=1000'
assert '_V07D2_LAMBDA_RL    = 0.0' in s23,         'LAMBDA_RL=0'
assert "pure IL mode" in s23,                      'pure IL comment'
assert "'1'))   # v07d21: pure IL" in s23,         'GAMES_PER_ITER=1'
assert '_V07D2_LAMBDA_RANK  = 0.5' in s23,         'LAMBDA_RANK kept'
assert '_il_won_ep' in s23,                        'LambdaRank won tracking kept'
assert 'F.logsigmoid' in s23,                      'LambdaRank loss kept'
assert '_v07d21_ws_path' in s23,                   'v07d21 warm start path'
assert 'v0-07d19-remote-pc' in s23,                'v07d19 model path'
assert '_v07d19_ws_path' not in s23,               'no old var name'
assert '_v07d2_sp_buffer' not in s23,              'no SP buffer (v07d20 removed)'
assert 'v0-07d21 Summary' in s23,                  'summary label'
assert 'Memory safety stop' in s23,                'memory safety stop'
print('All sanity checks passed.')
