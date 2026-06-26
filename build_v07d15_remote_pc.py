"""Build pokemon-20260625-v0-07d15-remote-pc.ipynb from v07d13-remote-pc notebook.

v07d15: conservative — memory fix + speedups (no algorithmic changes).

Changes from v07d13:
  [MEMORY FIX]
    1. Remove episode buffer accumulation (_ep_feats/.extend() × n_iters → OOM at 150 iters).
       PPO update uses _ro (current iter only); accumulated buffer was cross-version reuse
       artifact from offline PPO era. Now obsolete.
    2. del _ro, _adv, _ret + gc.collect() after each iter's PPO update → release memory immediately.

  [SPEEDUP]
    3. torch.no_grad() in _v07d2_collect model inference → skip autograd graph build
       during game collection (gradients not needed; pure inference).
    4. IL batch data prep moved from minibatch loop to epoch level:
       was 4 epochs × ~10 minibatches = 40x per iter (np.random.choice + iterrows + concatenate);
       now 4x per iter → ~10x speedup for IL batch construction.

N_ITERS stays at 100 (same as v07d13) to validate memory fix and measure speedup.
Expected: similar winner_margin_inference to v07d13 (+0.0269) but lower peak RAM
and shorter wall time. If validated, v07d16 can push to 150-200 iters.
"""
import json

SRC = '/kaggle/working/pokemon-20260625-v0-07d13-remote-pc.ipynb'
DST = '/kaggle/working/pokemon-20260625-v0-07d15-remote-pc.ipynb'

with open(SRC) as f:
    nb = json.load(f)

cells = nb['cells']

# ── Cell[1]: Config ───────────────────────────────────────────────────────────
src1 = ''.join(cells[1]['source'])
src1 = src1.replace(
    "EXPERIMENT_NAME   = 'v0_07d13_n_iters_100'",
    "EXPERIMENT_NAME   = 'v0_07d15_memory_speedup'"
)
src1 = src1.replace(
    "RUN_PREFIX = os.environ.get('V07D13_RUN_PREFIX', 'pokemon-20260625-v0-07d13-remote-pc')",
    "RUN_PREFIX = os.environ.get('V07D15_RUN_PREFIX', 'pokemon-20260625-v0-07d15-remote-pc')"
)
assert 'v0_07d15_memory_speedup' in src1, 'EXPERIMENT_NAME'
assert 'v0-07d15-remote-pc' in src1, 'RUN_PREFIX'
cells[1]['source'] = src1.splitlines(keepends=True)

# ── Cell[23]: RL training ─────────────────────────────────────────────────────
src23 = ''.join(cells[23]['source'])

# ── Patch 1: torch.no_grad() in collect model inference ──────────────────────
OLD_COLLECT_INFER = """\
                        X  = _v07d2_live_features(obs, opts_r)
                        Xt = torch.tensor(X, dtype=torch.float32, device=device)
                        logits, val = model(Xt)
                        lp_all  = F.log_softmax(logits, dim=0)
                        act_idx = Categorical(logits=logits).sample()
                        lp      = float(lp_all[act_idx].item())"""
NEW_COLLECT_INFER = """\
                        X  = _v07d2_live_features(obs, opts_r)
                        Xt = torch.tensor(X, dtype=torch.float32, device=device)
                        with torch.no_grad():  # v07d15: skip autograd during collection
                            logits, val = model(Xt)
                            lp_all  = F.log_softmax(logits, dim=0)
                            act_idx = Categorical(logits=logits).sample()
                        lp      = float(lp_all[act_idx].item())"""
assert OLD_COLLECT_INFER in src23, 'collect no_grad patch target not found'
src23 = src23.replace(OLD_COLLECT_INFER, NEW_COLLECT_INFER)

# ── Patch 2: Remove episode buffer init ──────────────────────────────────────
OLD_EP_INIT = """\
        # ── Episode buffer (accumulated across all iters for cross-version reuse) ──
        _ep_feats, _ep_nopts, _ep_acts, _ep_lps = [], [], [], []
        _ep_vals, _ep_srews, _ep_advs, _ep_rets, _ep_iters = [], [], [], [], []"""
NEW_EP_INIT = """\
        # v07d15: episode buffer removed (was O(n_iters) memory → OOM at 150 iters)
        _ep_total_decisions = 0"""
assert OLD_EP_INIT in src23, 'episode buffer init not found'
src23 = src23.replace(OLD_EP_INIT, NEW_EP_INIT)

# ── Patch 3: Remove extend calls + del _ro + gc.collect() ────────────────────
OLD_EXTEND = """\
            _ep_feats.extend(_ro['feats']); _ep_nopts.extend(_ro['n_opts'])
            _ep_acts.extend(_ro['actions']); _ep_lps.extend(_ro['log_probs'])
            _ep_vals.extend(_ro['values']); _ep_srews.extend(_ro['step_rewards'])
            _ep_advs.extend(_adv.tolist()); _ep_rets.extend(_ret.tolist())
            _ep_iters.extend([_rl_iter] * len(_ro['actions']))
            _m         = _v07d2_update(_ro, _adv, _ret, _v07d2_device)
            _vg        = [r for r in _ro['results'] if r['result'] is not None]
            _wr        = float(np.mean([r['won'] for r in _vg])) if _vg else 0.0
            _nmain     = sum(r['n_main'] for r in _ro['results'])
            _elapsed   = _v07d2_time.time() - _t0
            _oc        = _ro['opp_counts']"""
NEW_EXTEND = """\
            _m         = _v07d2_update(_ro, _adv, _ret, _v07d2_device)
            _vg        = [r for r in _ro['results'] if r['result'] is not None]
            _wr        = float(np.mean([r['won'] for r in _vg])) if _vg else 0.0
            _nmain     = sum(r['n_main'] for r in _ro['results'])
            _elapsed   = _v07d2_time.time() - _t0
            _oc        = _ro['opp_counts']
            _ep_total_decisions += _nmain  # v07d15: track count only (buffer removed)
            del _ro, _adv, _ret           # v07d15: release memory immediately
            __import__('gc').collect()"""
assert OLD_EXTEND in src23, 'extend block not found'
src23 = src23.replace(OLD_EXTEND, NEW_EXTEND)

# ── Patch 4: Episode save stub ────────────────────────────────────────────────
OLD_EP_SAVE = """\
        # ── Save episode buffer for cross-version reuse ───────────────────────────
        _ep_path = OUTPUT_DIR / (RUN_PREFIX + '-rl_episodes.pt')
        torch.save({
            'feats': _ep_feats, 'n_opts': _ep_nopts, 'actions': _ep_acts,
            'log_probs_beh': _ep_lps, 'values_beh': _ep_vals,
            'step_rewards': _ep_srews, 'adv': _ep_advs, 'ret': _ep_rets,
            'iter_idx': _ep_iters,
            'meta': {
                'behavioral_version': RUN_PREFIX,
                'n_iters': _V07D2_N_ITERS, 'games_per_iter': _V07D2_GAMES_PER_ITER,
                'total_decisions': len(_ep_acts),
                'prize_step_coef': _V07D2_PRIZE_STEP_COEF,
                'lambda_rl': _V07D2_LAMBDA_RL, 'lambda_il': _V07D2_LAMBDA_IL,
                'opponent_mix': '35pct_rule/30pct_IL/35pct_self',
            },
        }, str(_ep_path))
        ARTIFACT_PATHS['v07d3_rl_episodes'] = str(_ep_path)
        print(f'Episode buffer saved: {len(_ep_acts)} decisions → {_ep_path.name}')"""
NEW_EP_SAVE = """\
        # v07d15: episode buffer disabled; save metadata stub only
        _ep_path = OUTPUT_DIR / (RUN_PREFIX + '-rl_episodes.pt')
        torch.save({
            'feats': [], 'n_opts': [], 'actions': [],
            'log_probs_beh': [], 'values_beh': [],
            'step_rewards': [], 'adv': [], 'ret': [],
            'iter_idx': [],
            'meta': {
                'behavioral_version': RUN_PREFIX,
                'n_iters': _V07D2_N_ITERS, 'games_per_iter': _V07D2_GAMES_PER_ITER,
                'total_decisions': _ep_total_decisions,
                'prize_step_coef': _V07D2_PRIZE_STEP_COEF,
                'lambda_rl': _V07D2_LAMBDA_RL, 'lambda_il': _V07D2_LAMBDA_IL,
                'opponent_mix': '35pct_rule/30pct_IL/35pct_self',
                'note': 'v07d15: buffer disabled (memory fix)',
            },
        }, str(_ep_path))
        ARTIFACT_PATHS['v07d3_rl_episodes'] = str(_ep_path)
        print(f'Episode buffer stub (memory fix): {_ep_total_decisions} decisions → {_ep_path.name}')"""
assert OLD_EP_SAVE in src23, 'episode save block not found'
src23 = src23.replace(OLD_EP_SAVE, NEW_EP_SAVE)

# ── Patch 5: IL batch prep moved from minibatch loop to epoch level ───────────
# Step 5a: inject epoch-level IL prep after random.shuffle, before minibatch loop
OLD_EPOCH_HEADER = """\
            random.shuffle(idxs)
            for st in range(0, n, _V07D2_MINIBATCH):"""
NEW_EPOCH_HEADER = """\
            random.shuffle(idxs)
            # v07d15: IL data prep once per epoch (was once per minibatch — ~10x speedup)
            _il_idxs_ep = np.random.choice(_train_idxs_v7d2, size=_V07D2_IL_BATCH, replace=False)
            _il_rows_ep = _train_df_v7d2.iloc[_il_idxs_ep]
            _il_Xp_ep, _il_off_ep, _il_ch_ep = [], [0], []
            for _, _il_r_ep in _il_rows_ep.iterrows():
                _s_ep = int(_il_r_ep['option_start']); _e_ep = int(_il_r_ep['option_end'])
                if _e_ep - _s_ep >= 2:
                    _il_Xp_ep.append(_X_full_v7d2[_s_ep:_e_ep])
                    _il_off_ep.append(_il_off_ep[-1] + (_e_ep - _s_ep))
                    _il_ch_ep.append(int(_il_r_ep['chosen_index']))
            _il_Xt_ep = (torch.tensor(np.concatenate(_il_Xp_ep), dtype=torch.float32, device=device)
                         if _il_Xp_ep else None)
            for st in range(0, n, _V07D2_MINIBATCH):"""
assert OLD_EPOCH_HEADER in src23, 'epoch header not found'
src23 = src23.replace(OLD_EPOCH_HEADER, NEW_EPOCH_HEADER)

# Step 5b: replace IL data prep + forward inside minibatch loop with epoch-tensor forward
OLD_IL_INNER = """\
                # IL forward (replay data, separate mini-batch)
                il_idxs = np.random.choice(_train_idxs_v7d2, size=_V07D2_IL_BATCH, replace=False)
                il_rows = _train_df_v7d2.iloc[il_idxs]
                il_X_parts, il_offsets, il_chosen = [], [0], []
                for _, il_row in il_rows.iterrows():
                    s_il = int(il_row['option_start']); e_il = int(il_row['option_end'])
                    if e_il - s_il >= 2:
                        il_X_parts.append(_X_full_v7d2[s_il:e_il])
                        il_offsets.append(il_offsets[-1] + (e_il - s_il))
                        il_chosen.append(int(il_row['chosen_index']))

                if il_X_parts:
                    Xt_il     = torch.tensor(np.concatenate(il_X_parts), dtype=torch.float32, device=device)
                    logits_il = _v07d2_model.policy(Xt_il)
                    il_loss   = torch.zeros(1, device=device)
                    for k, ck in enumerate(il_chosen):
                        lo, hi = il_offsets[k], il_offsets[k + 1]
                        il_loss = il_loss + F.cross_entropy(
                            logits_il[lo:hi].unsqueeze(0),
                            torch.tensor([ck], device=device))
                    il_loss = il_loss / len(il_chosen)
                else:
                    il_loss = torch.zeros(1, device=device)"""
NEW_IL_INNER = """\
                # IL forward (data prep at epoch level — v07d15 speedup)
                if _il_Xt_ep is not None:
                    logits_il = _v07d2_model.policy(_il_Xt_ep)
                    il_loss   = torch.zeros(1, device=device)
                    for k, ck in enumerate(_il_ch_ep):
                        lo, hi = _il_off_ep[k], _il_off_ep[k + 1]
                        il_loss = il_loss + F.cross_entropy(
                            logits_il[lo:hi].unsqueeze(0),
                            torch.tensor([ck], device=device))
                    il_loss = il_loss / len(_il_ch_ep)
                else:
                    il_loss = torch.zeros(1, device=device)"""
assert OLD_IL_INNER in src23, 'IL inner block not found'
src23 = src23.replace(OLD_IL_INNER, NEW_IL_INNER)

# ── Patch 6: summary label ────────────────────────────────────────────────────
assert "print(f'\\n=== v0-07d13 Summary ===')" in src23, 'summary label'
src23 = src23.replace(
    "    print(f'\\n=== v0-07d13 Summary ===')",
    "    print(f'\\n=== v0-07d15 Summary ===')"
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

assert 'v0_07d15_memory_speedup' in s1,              'EXPERIMENT_NAME'
assert 'v0-07d15-remote-pc' in s1,                   'RUN_PREFIX'
assert '_V07D2_FEAT_DIM      = 96' in s23,            'FEAT_DIM=96'
assert "'100'))" in s23,                              'N_ITERS=100'
assert 'with torch.no_grad():  # v07d15' in s23,      'no_grad in collect'
assert '_ep_total_decisions = 0' in s23,              'ep buffer removed'
assert '_ep_feats.extend' not in s23,                 'no _ep_feats.extend accumulation'
assert '_ep_acts.extend' not in s23,                  'no extend calls'
assert 'del _ro, _adv, _ret' in s23,                  'del _ro after update'
assert '__import__(\'gc\').collect()' in s23,          'gc.collect'
assert '_il_Xt_ep' in s23,                            'epoch-level IL tensor'
assert 'v07d15: IL data prep once per epoch' in s23,  'epoch IL comment'
assert 'il_idxs = np.random.choice' not in s23,       'no per-minibatch IL prep'
assert 'v07d15: buffer disabled' in s23,              'ep save stub'
assert 'v0-07d15 Summary' in s23,                     'summary label'
assert '_V07D2_LAMBDA_IL    = 0.5' in s23,            'lambda_il=0.5'
assert '_v07d2_best_margin' in s23,                   'best ckpt tracking'
assert 'v07d12 saved model: wm=' in s23,              'measurement fix'
assert '_sv97' in s23,                                '97-dim padding'
print('All sanity checks passed.')
