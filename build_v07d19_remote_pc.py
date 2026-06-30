"""Build pokemon-20260625-v0-07d19-remote-pc.ipynb from v07d18-remote-pc notebook.

v07d19: aggressive — LambdaRank cross-decision pairwise loss (v05d25 inspired).

Changes from v07d18:
  [BIG WEAPON: LambdaRank IL]
    1. Add _V07D2_LAMBDA_RANK = 0.5 hyperparameter.
    2. Online PPO: track won label in IL batch (_il_won_ep).
       After CE loss, compute pairwise ranking loss:
         for every (winner decision, loser decision) pair,
         push winner_chosen_score > loser_chosen_score.
       This is v05d25 LambdaRank applied to the MLP.
    3. Offline PPO: same LambdaRank addition.
    4. total_loss += _V07D2_LAMBDA_RANK * _rank_loss (both paths).

  [COMPOUND LEARNING ROUND 4]
    5. Warm start from v07d18 best model (compound round 4).

Rationale:
  Current IL CE loss: within each decision, push chosen option score > others.
  New LambdaRank: across all decisions in batch, push winner chosen score > loser chosen score.
  This directly optimizes winner_margin = mean(winner_scores) - mean(loser_scores).
  v05d25 (strongest Kaggle submission) uses LambdaRank with winner_weight=4x.
  This brings the same philosophy to the MLP+RL framework.
"""
import json

SRC = '/kaggle/working/pokemon-20260625-v0-07d18-remote-pc.ipynb'
DST = '/kaggle/working/pokemon-20260625-v0-07d19-remote-pc.ipynb'

with open(SRC) as f:
    nb = json.load(f)

cells = nb['cells']

# ── Cell[1]: Config ───────────────────────────────────────────────────────────
src1 = ''.join(cells[1]['source'])
src1 = src1.replace(
    "EXPERIMENT_NAME   = 'v0_07d18_compound_warmstart_r3'",
    "EXPERIMENT_NAME   = 'v0_07d19_lambdarank_il'"
)
src1 = src1.replace(
    "RUN_PREFIX = os.environ.get('V07D18_RUN_PREFIX', 'pokemon-20260625-v0-07d18-remote-pc')",
    "RUN_PREFIX = os.environ.get('V07D19_RUN_PREFIX', 'pokemon-20260625-v0-07d19-remote-pc')"
)
assert 'v0_07d19_lambdarank_il' in src1, 'EXPERIMENT_NAME'
assert 'v0-07d19-remote-pc' in src1, 'RUN_PREFIX'
cells[1]['source'] = src1.splitlines(keepends=True)

# ── Cell[23]: RL training ─────────────────────────────────────────────────────
src23 = ''.join(cells[23]['source'])

# ── Patch 1: Add _V07D2_LAMBDA_RANK hyperparameter ───────────────────────────
OLD_LAMBDA = \
    "    _V07D2_LAMBDA_IL    = 0.5  # v07d9: proven sweet spot (v07d7 learning_promote)"
NEW_LAMBDA = \
    "    _V07D2_LAMBDA_IL    = 0.5  # v07d9: proven sweet spot (v07d7 learning_promote)\n" \
    "    _V07D2_LAMBDA_RANK  = 0.5  # v07d19: LambdaRank cross-decision pairwise loss weight"
assert OLD_LAMBDA in src23, 'LAMBDA_IL line not found'
src23 = src23.replace(OLD_LAMBDA, NEW_LAMBDA)

# ── Patch 2: Online PPO — track won label in IL batch prep ────────────────────
OLD_ONLINE_PREP = \
    "            _il_Xp_ep, _il_off_ep, _il_ch_ep = [], [0], []\n" \
    "            for _, _il_r_ep in _il_rows_ep.iterrows():\n" \
    "                _s_ep = int(_il_r_ep['option_start']); _e_ep = int(_il_r_ep['option_end'])\n" \
    "                if _e_ep - _s_ep >= 2:\n" \
    "                    _il_Xp_ep.append(_X_full_v7d2[_s_ep:_e_ep])\n" \
    "                    _il_off_ep.append(_il_off_ep[-1] + (_e_ep - _s_ep))\n" \
    "                    _il_ch_ep.append(int(_il_r_ep['chosen_index']))"
NEW_ONLINE_PREP = \
    "            _il_Xp_ep, _il_off_ep, _il_ch_ep, _il_won_ep = [], [0], [], []  # v07d19: track won\n" \
    "            for _, _il_r_ep in _il_rows_ep.iterrows():\n" \
    "                _s_ep = int(_il_r_ep['option_start']); _e_ep = int(_il_r_ep['option_end'])\n" \
    "                if _e_ep - _s_ep >= 2:\n" \
    "                    _il_Xp_ep.append(_X_full_v7d2[_s_ep:_e_ep])\n" \
    "                    _il_off_ep.append(_il_off_ep[-1] + (_e_ep - _s_ep))\n" \
    "                    _il_ch_ep.append(int(_il_r_ep['chosen_index']))\n" \
    "                    _il_won_ep.append(bool(_il_r_ep['won']))  # v07d19: won label"
assert OLD_ONLINE_PREP in src23, 'Online IL batch prep not found'
src23 = src23.replace(OLD_ONLINE_PREP, NEW_ONLINE_PREP)

# ── Patch 3: Online PPO — add LambdaRank loss + update total_loss ─────────────
OLD_ONLINE_LOSS = \
    "                # IL forward (data prep at epoch level — v07d15 speedup)\n" \
    "                if _il_Xt_ep is not None:\n" \
    "                    logits_il = _v07d2_model.policy(_il_Xt_ep)\n" \
    "                    il_loss   = torch.zeros(1, device=device)\n" \
    "                    for k, ck in enumerate(_il_ch_ep):\n" \
    "                        lo, hi = _il_off_ep[k], _il_off_ep[k + 1]\n" \
    "                        il_loss = il_loss + F.cross_entropy(\n" \
    "                            logits_il[lo:hi].unsqueeze(0),\n" \
    "                            torch.tensor([ck], device=device))\n" \
    "                    il_loss = il_loss / len(_il_ch_ep)\n" \
    "                else:\n" \
    "                    il_loss = torch.zeros(1, device=device)\n" \
    "\n" \
    "                total_loss = _V07D2_LAMBDA_RL * ppo_loss + _V07D2_LAMBDA_IL * il_loss"
NEW_ONLINE_LOSS = \
    "                # IL forward (data prep at epoch level — v07d15 speedup)\n" \
    "                if _il_Xt_ep is not None:\n" \
    "                    logits_il = _v07d2_model.policy(_il_Xt_ep)\n" \
    "                    il_loss   = torch.zeros(1, device=device)\n" \
    "                    for k, ck in enumerate(_il_ch_ep):\n" \
    "                        lo, hi = _il_off_ep[k], _il_off_ep[k + 1]\n" \
    "                        il_loss = il_loss + F.cross_entropy(\n" \
    "                            logits_il[lo:hi].unsqueeze(0),\n" \
    "                            torch.tensor([ck], device=device))\n" \
    "                    il_loss = il_loss / len(_il_ch_ep)\n" \
    "                    # v07d19: LambdaRank — winner chosen score > loser chosen score (cross-decision)\n" \
    "                    _il_ch_sc  = torch.stack([logits_il[_il_off_ep[k] + ck] for k, ck in enumerate(_il_ch_ep)])\n" \
    "                    _il_w_mask = torch.tensor(_il_won_ep, dtype=torch.bool, device=device)\n" \
    "                    _il_w_sc, _il_l_sc = _il_ch_sc[_il_w_mask], _il_ch_sc[~_il_w_mask]\n" \
    "                    if len(_il_w_sc) > 0 and len(_il_l_sc) > 0:\n" \
    "                        _rank_loss = -F.logsigmoid(_il_w_sc.unsqueeze(1) - _il_l_sc.unsqueeze(0)).mean()\n" \
    "                    else:\n" \
    "                        _rank_loss = torch.zeros(1, device=device)\n" \
    "                else:\n" \
    "                    il_loss    = torch.zeros(1, device=device)\n" \
    "                    _rank_loss = torch.zeros(1, device=device)\n" \
    "\n" \
    "                total_loss = _V07D2_LAMBDA_RL * ppo_loss + _V07D2_LAMBDA_IL * il_loss + _V07D2_LAMBDA_RANK * _rank_loss"
assert OLD_ONLINE_LOSS in src23, 'Online IL loss block not found'
src23 = src23.replace(OLD_ONLINE_LOSS, NEW_ONLINE_LOSS)

# ── Patch 4: Offline PPO — track won label in IL batch prep ──────────────────
OLD_OFFLINE_PREP = \
    "                _il_Xp_of, _il_off_of, _il_ch_of = [], [0], []\n" \
    "                for _, _ir_of in _il_rows_of.iterrows():\n" \
    "                    _s_of = int(_ir_of['option_start']); _e_of = int(_ir_of['option_end'])\n" \
    "                    if _e_of - _s_of >= 2:\n" \
    "                        _il_Xp_of.append(_X_full_v7d2[_s_of:_e_of])\n" \
    "                        _il_off_of.append(_il_off_of[-1] + (_e_of - _s_of))\n" \
    "                        _il_ch_of.append(int(_ir_of['chosen_index']))"
NEW_OFFLINE_PREP = \
    "                _il_Xp_of, _il_off_of, _il_ch_of, _il_won_of = [], [0], [], []  # v07d19: track won\n" \
    "                for _, _ir_of in _il_rows_of.iterrows():\n" \
    "                    _s_of = int(_ir_of['option_start']); _e_of = int(_ir_of['option_end'])\n" \
    "                    if _e_of - _s_of >= 2:\n" \
    "                        _il_Xp_of.append(_X_full_v7d2[_s_of:_e_of])\n" \
    "                        _il_off_of.append(_il_off_of[-1] + (_e_of - _s_of))\n" \
    "                        _il_ch_of.append(int(_ir_of['chosen_index']))\n" \
    "                        _il_won_of.append(bool(_ir_of['won']))  # v07d19: won label"
assert OLD_OFFLINE_PREP in src23, 'Offline IL batch prep not found'
src23 = src23.replace(OLD_OFFLINE_PREP, NEW_OFFLINE_PREP)

# ── Patch 5: Offline PPO — add LambdaRank loss + update total_loss ───────────
OLD_OFFLINE_LOSS = \
    "                if _il_Xp_of:\n" \
    "                    _Xt_il_of = torch.tensor(np.concatenate(_il_Xp_of),\n" \
    "                                             dtype=torch.float32, device=_v07d2_device)\n" \
    "                    _lg_il_of = _v07d2_model.policy(_Xt_il_of)\n" \
    "                    _il_l_of  = torch.zeros(1, device=_v07d2_device)\n" \
    "                    for _k_il, _ck_il in enumerate(_il_ch_of):\n" \
    "                        _lo_il = _il_off_of[_k_il]; _hi_il = _il_off_of[_k_il + 1]\n" \
    "                        _il_l_of = _il_l_of + F.cross_entropy(\n" \
    "                            _lg_il_of[_lo_il:_hi_il].unsqueeze(0),\n" \
    "                            torch.tensor([_ck_il], device=_v07d2_device))\n" \
    "                    _il_l_of = _il_l_of / len(_il_ch_of)\n" \
    "                else:\n" \
    "                    _il_l_of = torch.zeros(1, device=_v07d2_device)\n" \
    "                (_V07D2_LAMBDA_RL * _ppo_l_of + _V07D2_LAMBDA_IL * _il_l_of).backward()"
NEW_OFFLINE_LOSS = \
    "                if _il_Xp_of:\n" \
    "                    _Xt_il_of = torch.tensor(np.concatenate(_il_Xp_of),\n" \
    "                                             dtype=torch.float32, device=_v07d2_device)\n" \
    "                    _lg_il_of = _v07d2_model.policy(_Xt_il_of)\n" \
    "                    _il_l_of  = torch.zeros(1, device=_v07d2_device)\n" \
    "                    for _k_il, _ck_il in enumerate(_il_ch_of):\n" \
    "                        _lo_il = _il_off_of[_k_il]; _hi_il = _il_off_of[_k_il + 1]\n" \
    "                        _il_l_of = _il_l_of + F.cross_entropy(\n" \
    "                            _lg_il_of[_lo_il:_hi_il].unsqueeze(0),\n" \
    "                            torch.tensor([_ck_il], device=_v07d2_device))\n" \
    "                    _il_l_of = _il_l_of / len(_il_ch_of)\n" \
    "                    # v07d19: LambdaRank — winner chosen score > loser chosen score (cross-decision)\n" \
    "                    _ch_sc_of  = torch.stack([_lg_il_of[_il_off_of[_k] + _ck] for _k, _ck in enumerate(_il_ch_of)])\n" \
    "                    _wm_of     = torch.tensor(_il_won_of, dtype=torch.bool, device=_v07d2_device)\n" \
    "                    _ws_of, _ls_of = _ch_sc_of[_wm_of], _ch_sc_of[~_wm_of]\n" \
    "                    if len(_ws_of) > 0 and len(_ls_of) > 0:\n" \
    "                        _rk_l_of = -F.logsigmoid(_ws_of.unsqueeze(1) - _ls_of.unsqueeze(0)).mean()\n" \
    "                    else:\n" \
    "                        _rk_l_of = torch.zeros(1, device=_v07d2_device)\n" \
    "                else:\n" \
    "                    _il_l_of = torch.zeros(1, device=_v07d2_device)\n" \
    "                    _rk_l_of = torch.zeros(1, device=_v07d2_device)\n" \
    "                (_V07D2_LAMBDA_RL * _ppo_l_of + _V07D2_LAMBDA_IL * _il_l_of + _V07D2_LAMBDA_RANK * _rk_l_of).backward()"
assert OLD_OFFLINE_LOSS in src23, 'Offline IL loss block not found'
src23 = src23.replace(OLD_OFFLINE_LOSS, NEW_OFFLINE_LOSS)

# ── Patch 6: Update warm start path v07d18 → v07d17 (v07d18 plateaued: 0.0484 < v07d17 0.0490)
OLD_WS = \
    "    _v07d18_ws_path = Path('/kaggle/working/pokemon-20260625-v0-07d17-remote-pc/pokemon-20260625-v0-07d17-remote-pc-main_option_scorer.pt')\n" \
    "    _il_state2_97 = torch.load(str(_v07d18_ws_path), map_location='cpu', weights_only=True)\n" \
    "    _il_state2 = {k: (v[:, :96] if k == 'net.0.weight' else v) for k, v in _il_state2_97.items()}\n" \
    "    print(f'v07d18: warm start from v07d17 iter110 (+0.0490); net.0.weight: {_il_state2[\"net.0.weight\"].shape}')"
NEW_WS = \
    "    _v07d19_ws_path = Path('/kaggle/working/pokemon-20260625-v0-07d17-remote-pc/pokemon-20260625-v0-07d17-remote-pc-main_option_scorer.pt')\n" \
    "    _il_state2_97 = torch.load(str(_v07d19_ws_path), map_location='cpu', weights_only=True)\n" \
    "    _il_state2 = {k: (v[:, :96] if k == 'net.0.weight' else v) for k, v in _il_state2_97.items()}\n" \
    "    print(f'v07d19: warm start from v07d17 best model iter110 (+0.0490); net.0.weight: {_il_state2[\"net.0.weight\"].shape}')"
assert OLD_WS in src23, 'v07d18 warm start block not found'
src23 = src23.replace(OLD_WS, NEW_WS)

# ── Patch 7: Update rl_report to log lambda_rank (2 sites, different indent) ──
OLD_REPORT_A = \
    "                'lambda_rl': _V07D2_LAMBDA_RL, 'lambda_il': _V07D2_LAMBDA_IL,"
NEW_REPORT_A = \
    "                'lambda_rl': _V07D2_LAMBDA_RL, 'lambda_il': _V07D2_LAMBDA_IL, 'lambda_rank': _V07D2_LAMBDA_RANK,"
assert OLD_REPORT_A in src23, 'report site A not found'
src23 = src23.replace(OLD_REPORT_A, NEW_REPORT_A, 1)

OLD_REPORT_B = \
    "        'lambda_rl': _V07D2_LAMBDA_RL, 'lambda_il': _V07D2_LAMBDA_IL,"
NEW_REPORT_B = \
    "        'lambda_rl': _V07D2_LAMBDA_RL, 'lambda_il': _V07D2_LAMBDA_IL, 'lambda_rank': _V07D2_LAMBDA_RANK,"
assert OLD_REPORT_B in src23, 'report site B not found'
src23 = src23.replace(OLD_REPORT_B, NEW_REPORT_B, 1)

# ── Patch 8: Summary label ────────────────────────────────────────────────────
assert "print(f'\\n=== v0-07d18 Summary ===')" in src23, 'summary label'
src23 = src23.replace(
    "    print(f'\\n=== v0-07d18 Summary ===')",
    "    print(f'\\n=== v0-07d19 Summary ===')"
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

assert 'v0_07d19_lambdarank_il' in s1,             'EXPERIMENT_NAME'
assert 'v0-07d19-remote-pc' in s1,                  'RUN_PREFIX'
assert '_V07D2_FEAT_DIM      = 96' in s23,           'FEAT_DIM=96'
assert "'1000'))  # v07d16" in s23,                  'N_ITERS=1000'
assert '_V07D2_LAMBDA_RANK  = 0.5' in s23,           'LAMBDA_RANK'
assert '_il_won_ep' in s23,                           'online won tracking'
assert '_il_won_of' in s23,                           'offline won tracking'
assert 'LambdaRank' in s23,                          'LambdaRank comment'
assert 'F.logsigmoid' in s23,                        'logsigmoid rank loss'
assert '_rank_loss' in s23,                          'online rank loss'
assert '_rk_l_of' in s23,                            'offline rank loss'
assert 'LAMBDA_RANK * _rank_loss' in s23,            'online total_loss'
assert 'LAMBDA_RANK * _rk_l_of' in s23,             'offline total_loss'
assert '_v07d19_ws_path' in s23,                     'v07d19 warm start path'
assert 'v0-07d17-remote-pc' in s23,                  'v07d17 model path in warm start'
assert '_v07d18_ws_path' not in s23,                 'no old var name'
assert 'lambda_rank' in s23,                         'lambda_rank in report'
assert 'v0-07d19 Summary' in s23,                    'summary label'
assert 'Memory safety stop' in s23,                  'memory safety stop'
assert '__import__(\'gc\').collect()' not in s23,    'gc.collect removed'
print('All sanity checks passed.')
