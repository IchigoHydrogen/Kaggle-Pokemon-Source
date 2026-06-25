"""Build pokemon-20260625-v0-07d12-remote-pc.ipynb from v07d11-remote-pc notebook.

v07d12: conservative — fix rl_report measurement timing bug.

In v07d11, winner_margin and winner_margin_inference in rl_report were computed
BEFORE best-checkpoint restore. The saved model (iter-40, wm=+0.0080) was better
than the reported value (+0.0046), causing confusion in the promotion decision.

Fix: after the best-ckpt restore block (if restore happened), recompute
_post_wt1/_post_lt1/_post_margin and _inf_wt1/_inf_lt1/_inf_margin so the rl_report
accurately reflects the quality of the SAVED model.

No changes to training logic, algorithm, or hyperparameters.
"""
import json

SRC = '/kaggle/working/pokemon-20260625-v0-07d11-remote-pc.ipynb'
DST = '/kaggle/working/pokemon-20260625-v0-07d12-remote-pc.ipynb'

with open(SRC) as f:
    nb = json.load(f)

cells = nb['cells']

# ── Cell[1]: Config ───────────────────────────────────────────────────────────
src1 = ''.join(cells[1]['source'])

src1 = src1.replace(
    "EXPERIMENT_NAME   = 'v0_07d11_dim96_best_ckpt'",
    "EXPERIMENT_NAME   = 'v0_07d12_measurement_fix'"
)
src1 = src1.replace(
    "RUN_PREFIX = os.environ.get('V07D11_RUN_PREFIX', 'pokemon-20260625-v0-07d11-remote-pc')",
    "RUN_PREFIX = os.environ.get('V07D12_RUN_PREFIX', 'pokemon-20260625-v0-07d12-remote-pc')"
)
assert 'v0_07d12_measurement_fix' in src1, 'EXPERIMENT_NAME'
assert 'v0-07d12-remote-pc' in src1,       'RUN_PREFIX'
cells[1]['source'] = src1.splitlines(keepends=True)

# ── Cell[23]: RL training ─────────────────────────────────────────────────────
src23 = ''.join(cells[23]['source'])

# Fix: After best-ckpt restore, recompute _post_margin and _inf_margin.
# Insert after the else clause (end of best-ckpt block), before torch.save.
OLD_AFTER_CKPT = """\
        else:
            print(f'v07d11: final model kept (final={_final_wm:.4f} >= best={_v07d2_best_margin:.4f})')
    torch.save(_sv, str(_v07d2_model_path))"""
NEW_AFTER_CKPT = """\
        else:
            print(f'v07d11: final model kept (final={_final_wm:.4f} >= best={_v07d2_best_margin:.4f})')
    # v07d12: recompute post-RL metrics on SAVED model (after possible best-ckpt restore)
    _post_wt1, _post_lt1, _post_margin = _v07d2_winner_margin(
        _v07d2_model, _X_full_v7d2, _holdout_df_v7d2, _v07d2_device)
    _inf_wt1, _inf_lt1, _inf_margin = _v07d2_winner_margin_inference(
        _v07d2_model, _X_full_v7d2, _holdout_df_v7d2, _v07d2_device)
    print(f'v07d12 saved model: wm={_post_margin:.4f} wm_inf={_inf_margin:.4f}')
    torch.save(_sv, str(_v07d2_model_path))"""
assert OLD_AFTER_CKPT in src23, 'after-ckpt insertion point not found'
src23 = src23.replace(OLD_AFTER_CKPT, NEW_AFTER_CKPT)

# Update summary label
assert "print(f'\\n=== v0-07d11 Summary ===')" in src23, 'summary label'
src23 = src23.replace(
    "    print(f'\\n=== v0-07d11 Summary ===')",
    "    print(f'\\n=== v0-07d12 Summary ===')"
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

assert 'v0_07d12_measurement_fix' in s1,              'EXPERIMENT_NAME'
assert 'v0-07d12-remote-pc' in s1,                    'RUN_PREFIX'
assert '_V07D2_FEAT_DIM      = 96' in s23,             'FEAT_DIM=96'
assert '_v07d2_best_margin' in s23,                    'best ckpt init'
assert 'v07d12 saved model: wm=' in s23,               'measurement fix'
assert 'v07d12: recompute post-RL metrics' in s23,     'measurement fix comment'
assert '_sv97' in s23,                                 '97-dim padding kept'
assert 'new_zeros' in s23,                             '97-dim zero pad kept'
assert 'v0-07d12 Summary' in s23,                      'summary label'
assert 'np.zeros((n, _V07D2_FEAT_DIM)' in s23,         'Bug1 fix kept'
assert '_V07D2_LAMBDA_IL    = 0.5' in s23,             'lambda_il=0.5'
print('All sanity checks passed.')
