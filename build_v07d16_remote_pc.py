"""Build pokemon-20260625-v0-07d16-remote-pc.ipynb from v07d15-remote-pc notebook.

v07d16: aggressive — quality-weighted IL sampling + N_ITERS=1000 with early stopping.

Changes from v07d15:
  [WEIGHTED IL]
    1. Quality-weighted IL batch sampling (rank_bucket × won):
       top50+win=4.0, top50+lose=2.0, top200+win=3.0, top200+lose=1.0
       Vectorized np.where computation (O(1) vs iterrows).
       Effect: winner decisions go from 47% → 74% of each IL batch.
       Note: T200_T200 episode tier unavailable (v06d18 uses older episode dataset,
             JSONs not present in current input); rank_bucket is best available proxy.

  [CONVERGENCE]
    2. N_ITERS=1000 (was 100) with patience-based early stopping (patience=150 iters).
       Stops when no improvement in 150 consecutive iters; min 50 iters before eligibility.
       Memory safety hard-stop at 58 GB host RAM (4 GB headroom before 62 GB ceiling).

  [SPEEDUP]
    3. Remove gc.collect() per iter (was __import__('gc').collect() — adds ~2.7 min/100 iters;
       del _ro alone is sufficient for memory release).

Self-review notes:
  - _V07D2_PATIENCE added to BOTH smoke and normal branches (smoke gets 9999) to
    prevent NameError in the checkpoint block which runs every 10 iters even in smoke mode.
  - _rl_iters_run NOT tracked separately; _V07D2_N_ITERS kept in report fields.
    Actual early-stop iter is visible from len(rl_history) in rl_report.json.
  - IL probs print uses .sum() on boolean Series (no int() wrapping).
"""
import json

SRC = '/kaggle/working/pokemon-20260625-v0-07d15-remote-pc.ipynb'
DST = '/kaggle/working/pokemon-20260625-v0-07d16-remote-pc.ipynb'

with open(SRC) as f:
    nb = json.load(f)

cells = nb['cells']

# ── Cell[1]: Config ───────────────────────────────────────────────────────────
src1 = ''.join(cells[1]['source'])
src1 = src1.replace(
    "EXPERIMENT_NAME   = 'v0_07d15_memory_speedup'",
    "EXPERIMENT_NAME   = 'v0_07d16_il_weighted_1000iter'"
)
src1 = src1.replace(
    "RUN_PREFIX = os.environ.get('V07D15_RUN_PREFIX', 'pokemon-20260625-v0-07d15-remote-pc')",
    "RUN_PREFIX = os.environ.get('V07D16_RUN_PREFIX', 'pokemon-20260625-v0-07d16-remote-pc')"
)
# v07d16: allow smoke test to skip pipeline via SKIP_PIPELINE=1 env var
src1 = src1.replace(
    "SKIP_PIPELINE      = False          # v07d5: full pipeline with episodes-06-24",
    "SKIP_PIPELINE      = os.environ.get('SKIP_PIPELINE', 'false').lower() in ('true', '1')  # v07d16: configurable"
)
assert 'v0_07d16_il_weighted_1000iter' in src1, 'EXPERIMENT_NAME'
assert 'v0-07d16-remote-pc' in src1, 'RUN_PREFIX'
assert 'SKIP_PIPELINE' in src1 and 'os.environ.get' in src1, 'SKIP_PIPELINE env var'
cells[1]['source'] = src1.splitlines(keepends=True)

# ── Cell[23]: RL training ─────────────────────────────────────────────────────
src23 = ''.join(cells[23]['source'])

# ── Patch 1: N_ITERS=1000 + PATIENCE in both smoke and normal branches ────────
# Replace the entire smoke/normal N_ITERS block in one shot to avoid ordering issues.
OLD_NITERS_BLOCK = """\
        if RUN_MODE == 'smoke':
            _V07D2_N_ITERS        = int(os.environ.get('V07D2_N_ITERS', '2'))
            _V07D2_GAMES_PER_ITER = int(os.environ.get('V07D2_GAMES_PER_ITER', '10'))
        else:
            _V07D2_N_ITERS        = int(os.environ.get('V07D2_N_ITERS',        '100'))  # v07d13: extended from 50
            _V07D2_GAMES_PER_ITER = int(os.environ.get('V07D2_GAMES_PER_ITER', '200'))
        print(f'Hybrid IL+RL: {_V07D2_N_ITERS} iters x {_V07D2_GAMES_PER_ITER} games/iter  '"""
NEW_NITERS_BLOCK = """\
        if RUN_MODE == 'smoke':
            _V07D2_N_ITERS        = int(os.environ.get('V07D2_N_ITERS', '2'))
            _V07D2_GAMES_PER_ITER = int(os.environ.get('V07D2_GAMES_PER_ITER', '10'))
            _V07D2_PATIENCE       = 9999  # v07d16: no early stop in smoke mode
        else:
            _V07D2_N_ITERS        = int(os.environ.get('V07D2_N_ITERS',        '1000'))  # v07d16: max iters (early stop)
            _V07D2_GAMES_PER_ITER = int(os.environ.get('V07D2_GAMES_PER_ITER', '200'))
            _V07D2_PATIENCE       = int(os.environ.get('V07D2_PATIENCE',       '150'))   # v07d16: early stopping patience
        print(f'Hybrid IL+RL: {_V07D2_N_ITERS} iters x {_V07D2_GAMES_PER_ITER} games/iter  '"""
assert OLD_NITERS_BLOCK in src23, 'N_ITERS block not found'
src23 = src23.replace(OLD_NITERS_BLOCK, NEW_NITERS_BLOCK)

# ── Patch 2: IL quality weights (vectorized, after _train_idxs_v7d2) ──────────
OLD_TRAIN_IDX = """\
    _train_idxs_v7d2 = np.arange(len(_train_df_v7d2))
    print(f'Holdout: {len(_holdout_df_v7d2)}  Train (IL): {len(_train_df_v7d2)}')"""
NEW_TRAIN_IDX = """\
    _train_idxs_v7d2 = np.arange(len(_train_df_v7d2))
    print(f'Holdout: {len(_holdout_df_v7d2)}  Train (IL): {len(_train_df_v7d2)}')
    # v07d16: quality-weighted IL sampling (rank_bucket x won; top50~T200_T200 proxy)
    _il_weights_v7d2 = np.where(
        _train_df_v7d2['rank_bucket'].values == 'top50',
        np.where(_train_df_v7d2['won'].values, 4.0, 2.0),
        np.where(_train_df_v7d2['won'].values, 3.0, 1.0),
    ).astype(np.float32)
    _il_probs_v7d2 = _il_weights_v7d2 / _il_weights_v7d2.sum()
    print(f'IL weights: '
          f'top50+win={((_train_df_v7d2["rank_bucket"]=="top50") & _train_df_v7d2["won"]).sum()}(w=4) '
          f'top50+lose={((_train_df_v7d2["rank_bucket"]=="top50") & ~_train_df_v7d2["won"]).sum()}(w=2) '
          f'top200+win={((_train_df_v7d2["rank_bucket"]=="top200") & _train_df_v7d2["won"]).sum()}(w=3) '
          f'top200+lose={((_train_df_v7d2["rank_bucket"]=="top200") & ~_train_df_v7d2["won"]).sum()}(w=1)')"""
assert OLD_TRAIN_IDX in src23, '_train_idxs context not found'
src23 = src23.replace(OLD_TRAIN_IDX, NEW_TRAIN_IDX)

# ── Patch 3: Add no_improve counter init before RL loop ───────────────────────
OLD_BEST_INIT = """\
        _v07d2_best_margin = -float('inf')  # v07d11: best checkpoint tracking
        _v07d2_best_sd     = None
        for _rl_iter in range(_V07D2_N_ITERS):"""
NEW_BEST_INIT = """\
        _v07d2_best_margin = -float('inf')  # v07d11: best checkpoint tracking
        _v07d2_best_sd     = None
        _v07d2_no_improve  = 0              # v07d16: early stopping counter
        for _rl_iter in range(_V07D2_N_ITERS):"""
assert OLD_BEST_INIT in src23, 'best_margin init context not found'
src23 = src23.replace(OLD_BEST_INIT, NEW_BEST_INIT)

# ── Patch 4: Remove gc.collect() + add memory safety stop ─────────────────────
OLD_DEL_GC = """\
            del _ro, _adv, _ret           # v07d15: release memory immediately
            __import__('gc').collect()"""
NEW_DEL_GC = """\
            del _ro, _adv, _ret           # v07d15: release memory immediately
            # v07d16: memory safety stop (62GB host limit; 58GB = 4GB headroom)
            _host_ram_gb = __import__('psutil').Process().memory_info().rss / 1024**3
            if _host_ram_gb > 58.0:
                print(f'Memory safety stop @ iter {_rl_iter}: {_host_ram_gb:.1f}GB > 58GB')
                break"""
assert OLD_DEL_GC in src23, 'del gc block not found'
src23 = src23.replace(OLD_DEL_GC, NEW_DEL_GC)

# ── Patch 5: Weighted IL sampling (p=_il_probs_v7d2) ─────────────────────────
OLD_IL_CHOICE = """\
            _il_idxs_ep = np.random.choice(_train_idxs_v7d2, size=_V07D2_IL_BATCH, replace=False)"""
NEW_IL_CHOICE = """\
            _il_idxs_ep = np.random.choice(_train_idxs_v7d2, size=_V07D2_IL_BATCH, replace=False, p=_il_probs_v7d2)  # v07d16: quality-weighted"""
assert OLD_IL_CHOICE in src23, 'IL choice line not found'
src23 = src23.replace(OLD_IL_CHOICE, NEW_IL_CHOICE)

# ── Patch 6: Early stopping inside checkpoint eval block ──────────────────────
OLD_BEST_CKPT = """\
                if _iter_margin > _v07d2_best_margin:  # v07d11: track best ckpt
                    _v07d2_best_margin = _iter_margin
                    _v07d2_best_sd = {k: v.clone() for k, v in _v07d2_model.state_dict().items()}"""
NEW_BEST_CKPT = """\
                if _iter_margin > _v07d2_best_margin:  # v07d11: track best ckpt
                    _v07d2_best_margin = _iter_margin
                    _v07d2_best_sd = {k: v.clone() for k, v in _v07d2_model.state_dict().items()}
                    _v07d2_no_improve = 0
                else:
                    _v07d2_no_improve += 10
                # v07d16: early stopping (patience=150 iters, min 50 iters)
                if _v07d2_no_improve >= _V07D2_PATIENCE and _rl_iter >= 50:
                    print(f'Early stop @ iter {_rl_iter}: '
                          f'no improvement in {_v07d2_no_improve} iters '
                          f'(best={_v07d2_best_margin:.4f})')
                    break"""
assert OLD_BEST_CKPT in src23, 'best ckpt block not found'
src23 = src23.replace(OLD_BEST_CKPT, NEW_BEST_CKPT)

# ── Patch 7: Summary label ────────────────────────────────────────────────────
assert "print(f'\\n=== v0-07d15 Summary ===')" in src23, 'summary label'
src23 = src23.replace(
    "    print(f'\\n=== v0-07d15 Summary ===')",
    "    print(f'\\n=== v0-07d16 Summary ===')"
)

cells[23]['source'] = src23.splitlines(keepends=True)

# ── Clear outputs (code cells only — markdown cells must NOT have these fields) ──
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

assert 'v0_07d16_il_weighted_1000iter' in s1,          'EXPERIMENT_NAME'
assert 'v0-07d16-remote-pc' in s1,                     'RUN_PREFIX'
assert "os.environ.get('SKIP_PIPELINE'" in s1,          'SKIP_PIPELINE env var'
assert '_V07D2_FEAT_DIM      = 96' in s23,              'FEAT_DIM=96'
assert "'1000'))  # v07d16" in s23,                     'N_ITERS=1000'
assert '_V07D2_PATIENCE       = 9999' in s23,           'PATIENCE smoke mode'
assert "_V07D2_PATIENCE       = int(os.environ.get('V07D2_PATIENCE'" in s23, 'PATIENCE normal mode'
assert '_il_weights_v7d2' in s23,                       'IL weights computed'
assert '_il_probs_v7d2' in s23,                         'IL probs computed'
assert 'p=_il_probs_v7d2' in s23,                       'weighted IL sampling active'
assert '_v07d2_no_improve  = 0' in s23,                 'no_improve counter init'
assert '_v07d2_no_improve >= _V07D2_PATIENCE' in s23,   'early stopping check'
assert "__import__('gc').collect()" not in s23,         'gc.collect removed'
assert 'Memory safety stop' in s23,                     'memory safety stop'
assert 'with torch.no_grad():  # v07d15' in s23,        'no_grad in collect'
assert '_ep_feats.extend' not in s23,                   'no ep_feats accumulation'
assert 'del _ro, _adv, _ret' in s23,                    'del _ro after update'
assert '_V07D2_LAMBDA_IL    = 0.5' in s23,              'lambda_il=0.5'
assert '_v07d2_best_margin' in s23,                     'best ckpt tracking'
assert 'v07d12 saved model: wm=' in s23,                'measurement fix'
assert '_sv97' in s23,                                   '97-dim padding'
assert 'v0-07d16 Summary' in s23,                       'summary label'
print('All sanity checks passed.')
