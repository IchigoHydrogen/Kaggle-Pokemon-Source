"""Build pokemon-20260625-v0-07d11-remote-pc.ipynb from v07d10-remote-pc notebook.

v07d11: fix Part B (save 97-dim zero-padded model) + best-checkpoint saving.

Bug fixes vs v07d10:
  Bug 2 (Part B): Cell[25] builds submission tar.gz using _GUARD_INJECTION which
    defines _V06D15MainScorer(Linear(97,512)). The 96-dim RL model cannot load.
    Fix: after saving 96-dim RL model, re-save as 97-dim by zero-padding
    net.0.weight with one column. dim96 input gets zero weight → functionally 96-dim
    but structurally 97-dim for Part B compatibility.

  New: Best-checkpoint saving. iter40 margin (+0.0096) was better than final
    iter49 margin (-4.4e-05). Add tracking of best checkpoint across eval iters
    and restore before final model save.

Also: Remove the Cell[23] main.py file patch (harmless but ineffective — the
  _V06D15MainScorer class comes from _GUARD_INJECTION in Cell[25], not main.py).
"""
import json

SRC = '/kaggle/working/pokemon-20260625-v0-07d10-remote-pc.ipynb'
DST = '/kaggle/working/pokemon-20260625-v0-07d11-remote-pc.ipynb'

with open(SRC) as f:
    nb = json.load(f)

cells = nb['cells']

# ── Cell[1]: Config ───────────────────────────────────────────────────────────
src1 = ''.join(cells[1]['source'])

src1 = src1.replace(
    "EXPERIMENT_NAME   = 'v0_07d10_dim96_removal_fixed'",
    "EXPERIMENT_NAME   = 'v0_07d11_dim96_best_ckpt'"
)
src1 = src1.replace(
    "RUN_PREFIX = os.environ.get('V07D10_RUN_PREFIX', 'pokemon-20260625-v0-07d10-remote-pc')",
    "RUN_PREFIX = os.environ.get('V07D11_RUN_PREFIX', 'pokemon-20260625-v0-07d11-remote-pc')"
)
assert 'v0_07d11_dim96_best_ckpt' in src1, 'EXPERIMENT_NAME'
assert 'v0-07d11-remote-pc' in src1,       'RUN_PREFIX'
cells[1]['source'] = src1.splitlines(keepends=True)

# ── Cell[23]: RL training ─────────────────────────────────────────────────────
src23 = ''.join(cells[23]['source'])

# ── 1. Init best-checkpoint tracking before the training loop ─────────────────
OLD_LOOP_START = "        for _rl_iter in range(_V07D2_N_ITERS):"
NEW_LOOP_START = """\
        _v07d2_best_margin = -float('inf')  # v07d11: best checkpoint tracking
        _v07d2_best_sd     = None
        for _rl_iter in range(_V07D2_N_ITERS):"""
assert OLD_LOOP_START in src23, 'training loop start not found'
src23 = src23.replace(OLD_LOOP_START, NEW_LOOP_START)

# ── 2. Update best checkpoint after each checkpoint eval ─────────────────────
OLD_CKPT = """\
            if _rl_iter % 10 == 0 or _rl_iter == _V07D2_N_ITERS - 1:
                _iwt1, _ilt1, _iter_margin = _v07d2_winner_margin(
                    _v07d2_model, _X_full_v7d2, _holdout_df_v7d2, _v07d2_device)"""
NEW_CKPT = """\
            if _rl_iter % 10 == 0 or _rl_iter == _V07D2_N_ITERS - 1:
                _iwt1, _ilt1, _iter_margin = _v07d2_winner_margin(
                    _v07d2_model, _X_full_v7d2, _holdout_df_v7d2, _v07d2_device)
                if _iter_margin > _v07d2_best_margin:  # v07d11: track best ckpt
                    _v07d2_best_margin = _iter_margin
                    _v07d2_best_sd = {k: v.clone() for k, v in _v07d2_model.state_dict().items()}"""
assert OLD_CKPT in src23, 'checkpoint eval block not found'
src23 = src23.replace(OLD_CKPT, NEW_CKPT)

# ── 3. Load best checkpoint before model save (if better than final) ──────────
OLD_SAVE = "    torch.save(_sv, str(_v07d2_model_path))\n    print(f'model saved: {_v07d2_model_path.name}')"
NEW_SAVE = """\
    # v07d11: restore best checkpoint if better than final model
    if _v07d2_best_sd is not None:
        _, _, _final_wm = _v07d2_winner_margin(_v07d2_model, _X_full_v7d2, _holdout_df_v7d2, _v07d2_device)
        if _v07d2_best_margin > _final_wm:
            _v07d2_model.load_state_dict(_v07d2_best_sd)
            print(f'v07d11: best ckpt restored (best={_v07d2_best_margin:.4f} > final={_final_wm:.4f})')
            # recompute _sv from restored model
            _bp = dict(_v07d2_model.backbone.named_parameters())
            for _k in ['0', '2', '4']:
                _sv[f'net.{_k}.weight'] = _bp[f'{_k}.weight']
                _sv[f'net.{_k}.bias']   = _bp[f'{_k}.bias']
            _sv['net.6.weight'] = _v07d2_model.policy_head.weight
            _sv['net.6.bias']   = _v07d2_model.policy_head.bias
        else:
            print(f'v07d11: final model kept (final={_final_wm:.4f} >= best={_v07d2_best_margin:.4f})')
    torch.save(_sv, str(_v07d2_model_path))
    print(f'model saved: {_v07d2_model_path.name}')
    # v07d11: re-save as 97-dim (zero-pad dim96 col) for Part B compatibility
    _sv97 = {_k: (torch.cat([_v, _v.new_zeros(_v.shape[0], 1)], dim=1) if _k == 'net.0.weight' else _v) for _k, _v in _sv.items()}
    torch.save(_sv97, str(_v07d2_model_path))
    print(f'v07d11: re-saved as 97-dim (dim96 zeroed) for Part B compat; net.0.weight: {_sv97[\"net.0.weight\"].shape}')"""
assert OLD_SAVE in src23, 'model save line not found'
src23 = src23.replace(OLD_SAVE, NEW_SAVE)

# ── 4. Remove ineffective main.py file patch (kept harmless, but remove for clarity) ──
# The _V06D15MainScorer class comes from _GUARD_INJECTION in Cell[25], not main.py.
# Removing to avoid confusion.
OLD_MAINPY_PATCH = """\

    # ── v07d10: patch pipeline-generated main.py for 96-dim model ────────────
    _mainpy_path = OUTPUT_DIR / 'main.py'
    if _mainpy_path.exists():
        _mp = _mainpy_path.read_text()
        _mp_orig = _mp
        _mp = _mp.replace('nn.Linear(97, 512)', 'nn.Linear(96, 512)')
        _mp = _mp.replace('(n_opts, 97)', '(n_opts, 96)')
        # Remove X[:,96]=1.0 lines (return-condition leakage)
        import re as _re
        _mp = _re.sub(r"[ \\t]*X\\[:,\\s*96\\]\\s*=\\s*1\\.0[^\\n]*\\n", "", _mp)
        _mp = _re.sub(r"[ \\t]*X\\[\\s*:,96\\]\\s*=\\s*1\\.0[^\\n]*\\n", "", _mp)
        _mainpy_path.write_text(_mp)
        _changed = _mp != _mp_orig
        print(f'main.py patched for 96-dim: changed={_changed}')
    else:
        print('WARNING: main.py not found for patching')

"""
if OLD_MAINPY_PATCH in src23:
    src23 = src23.replace(OLD_MAINPY_PATCH, "\n")
    print('main.py patch removed')
else:
    print('WARNING: main.py patch block not found (skipping removal)')

# ── 5. Update summary label ───────────────────────────────────────────────────
assert "print(f'\\n=== v0-07d10 Summary ===')" in src23, 'summary label'
src23 = src23.replace(
    "    print(f'\\n=== v0-07d10 Summary ===')",
    "    print(f'\\n=== v0-07d11 Summary ===')"
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

assert 'v0_07d11_dim96_best_ckpt' in s1,             'EXPERIMENT_NAME'
assert 'v0-07d11-remote-pc' in s1,                   'RUN_PREFIX'
assert '_V07D2_FEAT_DIM      = 96' in s23,            'FEAT_DIM=96'
assert '_v07d2_best_margin' in s23,                   'best ckpt init'
assert '_v07d2_best_sd' in s23,                       'best ckpt sd'
assert 'v07d11: best ckpt restored' in s23,           'best ckpt restore'
assert '_sv97' in s23,                                '97-dim padding'
assert 'new_zeros' in s23,                            '97-dim zero pad'
assert 'v0-07d11 Summary' in s23,                     'summary label'
assert 'np.zeros((0, _V07D2_FEAT_DIM)' in s23,        'Bug1 fix kept'
assert 'np.zeros((n, _V07D2_FEAT_DIM)' in s23,        'Bug1 fix kept'
assert '_V07D2_LAMBDA_IL    = 0.5' in s23,            'lambda_il=0.5'
print('All sanity checks passed.')
