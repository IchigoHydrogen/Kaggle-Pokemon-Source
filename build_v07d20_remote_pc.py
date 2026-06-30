"""Build pokemon-20260625-v0-07d20-remote-pc.ipynb from v07d19-remote-pc notebook.

v07d20: aggressive — Self-Play IL Buffer (飛び道具 round 2).

Changes from v07d19:
  [BIG WEAPON: Self-Play IL Buffer]
    1. After each online game collection, extract winner decisions and add to
       _v07d2_sp_buffer (rolling buffer, max 2000 entries).
    2. In each IL batch: mix v06d18 data + up to IL_BATCH//2 SP winner decisions.
    3. SP entries always have won=True → LambdaRank sees more winner/loser pairs.

  [COMPOUND LEARNING]
    4. Warm start from v07d19 best model (+0.0494).

  LambdaRank (v07d19) retained.

Rationale:
  Current ceiling ~+0.049 is caused by static v06d18 IL data (4792 decisions).
  Self-play IL generates fresh winner data from the current model's own games.
  As the model improves, it wins more games → more high-quality SP winner data.
  Creates a positive feedback loop: better model → better SP data → better model.
  This is AlphaGo-style self-play applied to IL training.
"""
import json

SRC = '/kaggle/working/pokemon-20260625-v0-07d19-remote-pc.ipynb'
DST = '/kaggle/working/pokemon-20260625-v0-07d20-remote-pc.ipynb'

with open(SRC) as f:
    nb = json.load(f)

cells = nb['cells']

# ── Cell[1]: Config ───────────────────────────────────────────────────────────
src1 = ''.join(cells[1]['source'])
src1 = src1.replace(
    "EXPERIMENT_NAME   = 'v0_07d19_lambdarank_il'",
    "EXPERIMENT_NAME   = 'v0_07d20_selfplay_il_buffer'"
)
src1 = src1.replace(
    "RUN_PREFIX = os.environ.get('V07D19_RUN_PREFIX', 'pokemon-20260625-v0-07d19-remote-pc')",
    "RUN_PREFIX = os.environ.get('V07D20_RUN_PREFIX', 'pokemon-20260625-v0-07d20-remote-pc')"
)
assert 'v0_07d20_selfplay_il_buffer' in src1, 'EXPERIMENT_NAME'
assert 'v0-07d20-remote-pc' in src1, 'RUN_PREFIX'
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
    "    _v07d20_ws_path = Path('/kaggle/working/pokemon-20260625-v0-07d19-remote-pc/pokemon-20260625-v0-07d19-remote-pc-main_option_scorer.pt')\n" \
    "    _il_state2_97 = torch.load(str(_v07d20_ws_path), map_location='cpu', weights_only=True)\n" \
    "    _il_state2 = {k: (v[:, :96] if k == 'net.0.weight' else v) for k, v in _il_state2_97.items()}\n" \
    "    print(f'v07d20: warm start from v07d19 best model (+0.0494); net.0.weight: {_il_state2[\"net.0.weight\"].shape}')"
assert OLD_WS in src23, 'warm start block not found'
src23 = src23.replace(OLD_WS, NEW_WS)

# ── Patch 2: Add SP buffer init before main RL loop ──────────────────────────
OLD_LOOP_INIT = \
    "        _v07d2_best_margin = -float('inf')  # v07d11: best checkpoint tracking\n" \
    "        _v07d2_best_sd     = None\n" \
    "        _v07d2_no_improve  = 0              # v07d16: early stopping counter\n" \
    "        for _rl_iter in range(_V07D2_N_ITERS):"
NEW_LOOP_INIT = \
    "        _v07d2_best_margin = -float('inf')  # v07d11: best checkpoint tracking\n" \
    "        _v07d2_best_sd     = None\n" \
    "        _v07d2_no_improve  = 0              # v07d16: early stopping counter\n" \
    "        _v07d2_sp_buffer   = []             # v07d20: self-play IL buffer (winner decisions)\n" \
    "        _V07D2_SP_BUF_SIZE = 2000           # max winner decisions to keep\n" \
    "        for _rl_iter in range(_V07D2_N_ITERS):"
assert OLD_LOOP_INIT in src23, 'loop init block not found'
src23 = src23.replace(OLD_LOOP_INIT, NEW_LOOP_INIT)

# ── Patch 3: Populate SP buffer after each collection ────────────────────────
OLD_COLLECT = \
    "            _ro        = _v07d2_collect(_v07d2_model, _V07D2_GAMES_PER_ITER, _v07d2_device)\n" \
    "            _adv, _ret = _v07d2_gae(_ro)\n" \
    "            _m         = _v07d2_update(_ro, _adv, _ret, _v07d2_device)"
NEW_COLLECT = \
    "            _ro        = _v07d2_collect(_v07d2_model, _V07D2_GAMES_PER_ITER, _v07d2_device)\n" \
    "            # v07d20: add winner decisions to self-play IL buffer\n" \
    "            for _sp_k, _sp_gid in enumerate(_ro['game_ids']):\n" \
    "                if _ro['results'][_sp_gid]['won']:\n" \
    "                    _v07d2_sp_buffer.append((_ro['feats'][_sp_k], _ro['actions'][_sp_k]))\n" \
    "            if len(_v07d2_sp_buffer) > _V07D2_SP_BUF_SIZE:\n" \
    "                _v07d2_sp_buffer = _v07d2_sp_buffer[-_V07D2_SP_BUF_SIZE:]\n" \
    "            _adv, _ret = _v07d2_gae(_ro)\n" \
    "            _m         = _v07d2_update(_ro, _adv, _ret, _v07d2_device)"
assert OLD_COLLECT in src23, 'collect block not found'
src23 = src23.replace(OLD_COLLECT, NEW_COLLECT)

# ── Patch 4: Mix SP buffer into online IL batch (inside _v07d2_update) ────────
OLD_IL_TENSOR = \
    "            _il_Xt_ep = (torch.tensor(np.concatenate(_il_Xp_ep), dtype=torch.float32, device=device)\n" \
    "                         if _il_Xp_ep else None)\n" \
    "            for st in range(0, n, _V07D2_MINIBATCH):"
NEW_IL_TENSOR = \
    "            _il_Xt_ep = (torch.tensor(np.concatenate(_il_Xp_ep), dtype=torch.float32, device=device)\n" \
    "                         if _il_Xp_ep else None)\n" \
    "            # v07d20: mix self-play winner decisions into IL batch\n" \
    "            if len(_v07d2_sp_buffer) >= 16:\n" \
    "                _sp_n    = min(_V07D2_IL_BATCH // 2, len(_v07d2_sp_buffer))\n" \
    "                _sp_pick = np.random.choice(len(_v07d2_sp_buffer), size=_sp_n, replace=False)\n" \
    "                for _spi in _sp_pick:\n" \
    "                    _spX, _spa = _v07d2_sp_buffer[_spi]\n" \
    "                    if len(_spX) >= 2:\n" \
    "                        _il_Xp_ep.append(_spX)\n" \
    "                        _il_off_ep.append(_il_off_ep[-1] + len(_spX))\n" \
    "                        _il_ch_ep.append(_spa)\n" \
    "                        _il_won_ep.append(True)  # v07d20: SP = winner decisions\n" \
    "                if _il_Xp_ep:\n" \
    "                    _il_Xt_ep = torch.tensor(\n" \
    "                        np.concatenate(_il_Xp_ep), dtype=torch.float32, device=device)\n" \
    "            for st in range(0, n, _V07D2_MINIBATCH):"
assert OLD_IL_TENSOR in src23, 'IL tensor build block not found'
src23 = src23.replace(OLD_IL_TENSOR, NEW_IL_TENSOR)

# ── Patch 5: Summary label ────────────────────────────────────────────────────
assert "print(f'\\n=== v0-07d19 Summary ===')" in src23, 'summary label'
src23 = src23.replace(
    "    print(f'\\n=== v0-07d19 Summary ===')",
    "    print(f'\\n=== v0-07d20 Summary ===')"
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

assert 'v0_07d20_selfplay_il_buffer' in s1,          'EXPERIMENT_NAME'
assert 'v0-07d20-remote-pc' in s1,                    'RUN_PREFIX'
assert '_V07D2_FEAT_DIM      = 96' in s23,             'FEAT_DIM=96'
assert "'1000'))  # v07d16" in s23,                    'N_ITERS=1000'
assert '_V07D2_LAMBDA_RANK  = 0.5' in s23,             'LAMBDA_RANK kept'
assert '_v07d20_ws_path' in s23,                       'v07d20 warm start path'
assert 'v0-07d19-remote-pc' in s23,                    'v07d19 model path'
assert '_v07d19_ws_path' not in s23,                   'no old var name'
assert '_v07d2_sp_buffer   = []' in s23,               'SP buffer init'
assert '_V07D2_SP_BUF_SIZE = 2000' in s23,             'SP buffer size'
assert "_ro['results'][_sp_gid]['won']" in s23,        'SP buffer populate'
assert '_il_won_ep.append(True)' in s23,               'SP won label'
assert 'v07d20: SP = winner decisions' in s23,         'SP won comment'
assert '_il_Xt_ep = torch.tensor' in s23,              'SP tensor rebuild'
assert 'v0-07d20 Summary' in s23,                      'summary label'
assert 'Memory safety stop' in s23,                    'memory safety stop'
assert "F.logsigmoid" in s23,                          'LambdaRank retained'
print('All sanity checks passed.')
