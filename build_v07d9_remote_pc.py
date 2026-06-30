"""Build pokemon-20260625-v0-07d9-remote-pc.ipynb from v07d8-remote-pc notebook.

v07d9: aggressive — dim96 removal (feature matrix honest fix) + online PPO lambda_il=0.5
Changes vs v07d8:
  - dim96 (index 96): REMOVED from feature vector (_V07D2_FEAT_DIM: 97→96)
      * Feature matrix:  _X_full_v7d2[:, :96] after load (drop last column)
      * IL input weight: backbone.0.weight[:, :96] (drop dim96 connection)
      * Live features:   remove X[:, 96] = 1.0
      * Inference eval:  remove X_inf[:, 96] = 1.0 override (stored == inference now)
  - lambda_il: 0.1 → 0.5 (revert to proven sweet spot from v07d7)
  - EPISODE_SOURCE: None (online RL; feature dim changed → cannot reuse old episodes)
  - EXPERIMENT_NAME, RUN_PREFIX, summary label updated

  Hypothesis: dim96 = return-condition leakage (winner=1, loser=0 in training; always 1
  at inference). After removal, IL learns genuine board-state features with no shortcut.
  Stored wm == inference wm. RL on honest IL foundation can achieve higher winner_margin.

  Note: reward shaping (prize step rewards, coef=0.3) and iterative self-play were
  already implemented in v07d2 — no changes needed for those.
"""
import json

SRC = '/kaggle/working/pokemon-20260625-v0-07d8-remote-pc.ipynb'
DST = '/kaggle/working/pokemon-20260625-v0-07d9-remote-pc.ipynb'

with open(SRC) as f:
    nb = json.load(f)

cells = nb['cells']

# ── Cell[1]: Config ───────────────────────────────────────────────────────────
src1 = ''.join(cells[1]['source'])

src1 = src1.replace(
    "EXPERIMENT_NAME   = 'v0_07d8_online_ppo_lambda_il_01'",
    "EXPERIMENT_NAME   = 'v0_07d9_dim96_removal_online_ppo'"
)
src1 = src1.replace(
    "RUN_PREFIX = os.environ.get('V07D8_RUN_PREFIX', 'pokemon-20260625-v0-07d8-remote-pc')",
    "RUN_PREFIX = os.environ.get('V07D9_RUN_PREFIX', 'pokemon-20260625-v0-07d9-remote-pc')"
)

assert 'v0_07d9_dim96_removal_online_ppo' in src1, 'EXPERIMENT_NAME not updated'
assert 'v0-07d9-remote-pc' in src1, 'RUN_PREFIX not updated'
cells[1]['source'] = src1.splitlines(keepends=True)

# ── Cell[23]: RL training ─────────────────────────────────────────────────────
src23 = ''.join(cells[23]['source'])

# 1. FEAT_DIM: 97 → 96 (module-level constant)
assert '_V07D2_FEAT_DIM      = 97' in src23, 'FEAT_DIM=97 not found'
src23 = src23.replace(
    '_V07D2_FEAT_DIM      = 97',
    '_V07D2_FEAT_DIM      = 96   # v07d9: dim96 removed'
)

# 2. Drop dim96 column from loaded feature matrix (insert line after np.load)
OLD_LOAD = (
    '    _X_full_v7d2  = np.load(str(_il_feat_path2))\n'
    "    print(f'IL data: X={_X_full_v7d2.shape}, decisions={len(_main_df_v7d2)}')"
)
NEW_LOAD = (
    '    _X_full_v7d2  = np.load(str(_il_feat_path2))\n'
    '    _X_full_v7d2  = _X_full_v7d2[:, :96]          # v07d9: drop dim96 leakage column\n'
    "    print(f'IL data: X={_X_full_v7d2.shape}, decisions={len(_main_df_v7d2)}')"
)
assert OLD_LOAD in src23, 'feature matrix load block not found'
src23 = src23.replace(OLD_LOAD, NEW_LOAD)

# 3. IL weight loading: truncate backbone.0 (input layer) weight from [hidden,97] to [hidden,96]
OLD_WEIGHT_LOOP = (
    "        _ac_init2[f'{_bb}.weight'] = _il_state2[f'{_iln}.weight']\n"
    "        _ac_init2[f'{_bb}.bias']   = _il_state2[f'{_iln}.bias']"
)
NEW_WEIGHT_LOOP = (
    "        _w = _il_state2[f'{_iln}.weight']\n"
    "        if _li == 0:\n"
    "            _w = _w[:, :96]            # v07d9: drop dim96 column from input weights\n"
    "        _ac_init2[f'{_bb}.weight'] = _w\n"
    "        _ac_init2[f'{_bb}.bias']   = _il_state2[f'{_iln}.bias']"
)
assert OLD_WEIGHT_LOOP in src23, 'IL weight loop not found'
src23 = src23.replace(OLD_WEIGHT_LOOP, NEW_WEIGHT_LOOP)

# 4. Live features: remove X[:, 96] = 1.0 (return-condition column)
OLD_LIVE = (
    '        X[:, 96] = 1.0\n'
    '        return X'
)
NEW_LIVE = (
    '        # v07d9: dim96 removed — no return-condition column\n'
    '        return X'
)
assert OLD_LIVE in src23, 'X[:,96]=1.0 line not found'
src23 = src23.replace(OLD_LIVE, NEW_LIVE)

# 5. Inference eval: remove X_inf[:,96] override (stored == inference after dim96 removal)
OLD_INFER = (
    '    def _v07d2_winner_margin_inference(model, X_full, hdf, device):\n'
    '        # Inference-feature eval: dim96=1.0 for ALL (true inference; no win-label leakage)\n'
    '        X_inf = X_full.copy()\n'
    '        X_inf[:, 96] = 1.0\n'
    '        return _v07d2_winner_margin(model, X_inf, hdf, device)'
)
NEW_INFER = (
    '    def _v07d2_winner_margin_inference(model, X_full, hdf, device):\n'
    '        # v07d9: dim96 removed — stored and inference evals are now identical\n'
    '        return _v07d2_winner_margin(model, X_full, hdf, device)'
)
assert OLD_INFER in src23, 'inference eval function not found'
src23 = src23.replace(OLD_INFER, NEW_INFER)

# 6. lambda_il: 0.1 → 0.5
OLD_LAMBDA = "    _V07D2_LAMBDA_IL    = 0.1  # v07d8: extreme reduction (tests PPO clip stability at very low IL anchor)"
NEW_LAMBDA = "    _V07D2_LAMBDA_IL    = 0.5  # v07d9: proven sweet spot (v07d7 learning_promote)"
assert OLD_LAMBDA in src23, 'old lambda_il line not found'
src23 = src23.replace(OLD_LAMBDA, NEW_LAMBDA)

# 7. Summary label
assert "print(f'\\n=== v0-07d8 Summary ===')" in src23, 'summary label not found'
src23 = src23.replace(
    "    print(f'\\n=== v0-07d8 Summary ===')",
    "    print(f'\\n=== v0-07d9 Summary ===')"
)

cells[23]['source'] = src23.splitlines(keepends=True)

# ── Clear outputs ─────────────────────────────────────────────────────────────
for c in cells:
    c['outputs'] = []
    c['execution_count'] = None

with open(DST, 'w') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f'Written: {DST}')

# ── Sanity checks ─────────────────────────────────────────────────────────────
with open(DST) as f:
    nb2 = json.load(f)
s1  = ''.join(nb2['cells'][1]['source'])
s23 = ''.join(nb2['cells'][23]['source'])

assert 'v0_07d9_dim96_removal_online_ppo' in s1,   'EXPERIMENT_NAME'
assert 'v0-07d9-remote-pc' in s1,                  'RUN_PREFIX'
assert '_V07D2_FEAT_DIM      = 96' in s23,          'FEAT_DIM=96'
assert '_X_full_v7d2  = _X_full_v7d2[:, :96]' in s23, 'feature matrix slice'
assert "_w = _w[:, :96]" in s23,                   'IL weight truncation'
assert 'X[:, 96] = 1.0' not in s23,                'X[:,96]=1.0 removed'
assert 'X_inf[:, 96] = 1.0' not in s23,            'X_inf[:,96]=1.0 removed'
assert '_V07D2_LAMBDA_IL    = 0.5' in s23,          'lambda_il=0.5'
assert 'v0-07d9 Summary' in s23,                   'summary label'
assert 'EPISODE_SOURCE     = None' in s1,           'online RL'
print('All sanity checks passed.')
