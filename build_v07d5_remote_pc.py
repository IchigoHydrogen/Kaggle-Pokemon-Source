"""Build pokemon-20260625-v0-07d5-remote-pc.ipynb from v07d4 notebook."""
import json

SRC = '/kaggle/working/pokemon-20260625-v0-07d4.ipynb'
DST = '/kaggle/working/pokemon-20260625-v0-07d5-remote-pc.ipynb'

with open(SRC) as f:
    nb = json.load(f)

cells = nb['cells']

# ══════════════════════════════════════════════════════════════════════════════
# Cell[1]: Config
# ══════════════════════════════════════════════════════════════════════════════
src1 = ''.join(cells[1]['source'])

# 1. EXPERIMENT_NAME
src1 = src1.replace(
    "EXPERIMENT_NAME   = 'v0_07d4_offline_episode_training'",
    "EXPERIMENT_NAME   = 'v0_07d5_full_pipeline_new_data'"
)

# 2. SKIP_PIPELINE + remove EPISODE_SOURCE / LOAD_TRAINED_MODEL / EPISODE_WARMSTART
old_block = (
    "SKIP_PIPELINE     = os.environ.get('V07D4_SKIP_PIPELINE', 'true').lower() != 'false'\n"
    "EPISODE_SOURCE    = os.environ.get('V07D4_EPISODE_SOURCE',\n"
    "    '/kaggle/working/pokemon-20260625-v0-07d3/pokemon-20260625-v0-07d3-rl_episodes.pt')\n"
    "LOAD_TRAINED_MODEL = os.environ.get('V07D4_LOAD_TRAINED_MODEL', 'false') == 'true'\n"
    "EPISODE_WARMSTART = os.environ.get('V07D4_EPISODE_WARMSTART',\n"
    "    '/kaggle/working/pokemon-20260625-v0-07d3/pokemon-20260625-v0-07d3-main_option_scorer.pt')"
)
new_block = (
    "SKIP_PIPELINE      = False          # v07d5: full pipeline with episodes-06-24\n"
    "LOAD_TRAINED_MODEL = False\n"
    "EPISODE_SOURCE     = None           # v07d5: online RL, no pre-saved episodes\n"
    "EPISODE_WARMSTART  = None           # v07d5: IL initialization only"
)
assert old_block in src1, "old_block not found in Cell[1]"
src1 = src1.replace(old_block, new_block)

# 3. EPISODE_ROOT_CANDIDATES → episodes-2026-06-24 only
old_erc = (
    "EPISODE_ROOT_CANDIDATES = [\n"
    "    Path('/kaggle/input'),\n"
    "    Path('/kaggle/working'),\n"
    "    Path('.'),\n"
    "]"
)
new_erc = (
    "EPISODE_ROOT_CANDIDATES = [\n"
    "    Path('/kaggle/input/competitions/pokemon-tcg-ai-battle-episodes-2026-06-24'),\n"
    "    Path('/kaggle/working'),\n"
    "    Path('.'),\n"
    "]"
)
assert old_erc in src1, "old_erc not found in Cell[1]"
src1 = src1.replace(old_erc, new_erc)

# 4. TOP200_CSV_PATH → 06-24
src1 = src1.replace(
    "TOP200_CSV_PATH = Path('/kaggle/input/competitions/top200-20260622-ranking/top200-20260622-ranking.csv')",
    "TOP200_CSV_PATH = Path('/kaggle/input/competitions/top200-20260624-ranking/top200-20260624-ranking.csv')"
)

# 5. RUN_PREFIX
src1 = src1.replace(
    "RUN_PREFIX = os.environ.get('V05_RUN_PREFIX', 'pokemon-20260625-v0-07d4')",
    "RUN_PREFIX = os.environ.get('V07D5_RUN_PREFIX', 'pokemon-20260625-v0-07d5-remote-pc')"
)

cells[1]['source'] = src1.splitlines(keepends=True)

# ══════════════════════════════════════════════════════════════════════════════
# Cell[23]: RL training
# ══════════════════════════════════════════════════════════════════════════════
src23 = ''.join(cells[23]['source'])

# 1. Add _v07d2_winner_margin_inference after _v07d2_winner_margin
old_margin_fn_end = (
    "        wt1 = hw / max(1, tw);  lt1 = hl / max(1, tl)\n"
    "        return wt1, lt1, wt1 - lt1\n"
    "\n"
    "    _pre_wt1, _pre_lt1, _pre_margin = _v07d2_winner_margin("
)
new_margin_fn_end = (
    "        wt1 = hw / max(1, tw);  lt1 = hl / max(1, tl)\n"
    "        return wt1, lt1, wt1 - lt1\n"
    "\n"
    "    def _v07d2_winner_margin_inference(model, X_full, hdf, device):\n"
    "        # Inference-feature eval: dim96=1.0 for ALL (true inference; no win-label leakage)\n"
    "        X_inf = X_full.copy()\n"
    "        X_inf[:, 96] = 1.0\n"
    "        return _v07d2_winner_margin(model, X_inf, hdf, device)\n"
    "\n"
    "    _pre_wt1, _pre_lt1, _pre_margin = _v07d2_winner_margin("
)
assert old_margin_fn_end in src23, "old_margin_fn_end not found in Cell[23]"
src23 = src23.replace(old_margin_fn_end, new_margin_fn_end)

# 2. torch.manual_seed + _inf_* init before RL training section
# Line 597 has longer dashes — match exactly using the known suffix
old_rl_comment = "    # ── RL Training: online or offline (or load saved) ───────────────────────────"
assert old_rl_comment in src23, f"old_rl_comment not found: {repr(old_rl_comment[:60])}"
old_rl_start = (
    "    # ── RL Training: online or offline (or load saved) ───────────────────────────\n"
    "    _t_rl_start    = _v07d2_time.time()\n"
    "    _v07d2_history = []"
)
new_rl_start = (
    "    torch.manual_seed(RANDOM_SEED)  # v07d5: reproducible RL training\n"
    "    _inf_wt1 = _inf_lt1 = _inf_margin = None\n"
    "    # ── RL Training: online or offline (or load saved) ───────────────────────────\n"
    "    _t_rl_start    = _v07d2_time.time()\n"
    "    _v07d2_history = []"
)
assert old_rl_start in src23, "old_rl_start not found in Cell[23]"
src23 = src23.replace(old_rl_start, new_rl_start)

# 3. Add inference eval call after post-RL stored eval (online branch, lines 808-813)
old_post_eval = (
    "        _post_wt1, _post_lt1, _post_margin = _v07d2_winner_margin(\n"
    "            _v07d2_model, _X_full_v7d2, _holdout_df_v7d2, _v07d2_device)\n"
    "        print(f'Post-RL: wr={_post_wr:.3f}  wm={_post_margin:.4f} '\n"
    "              f'(wt1={_post_wt1:.4f} lt1={_post_lt1:.4f})')\n"
    "        print(f'Forgetting: IL_baseline={_pre_margin:.4f} -> RL={_post_margin:.4f} '\n"
    "              f'delta={_post_margin - _pre_margin:+.4f}')"
)
new_post_eval = (
    "        _post_wt1, _post_lt1, _post_margin = _v07d2_winner_margin(\n"
    "            _v07d2_model, _X_full_v7d2, _holdout_df_v7d2, _v07d2_device)\n"
    "        _inf_wt1, _inf_lt1, _inf_margin = _v07d2_winner_margin_inference(\n"
    "            _v07d2_model, _X_full_v7d2, _holdout_df_v7d2, _v07d2_device)\n"
    "        print(f'Post-RL (stored-feature):    wr={_post_wr:.3f}  wm={_post_margin:.4f} '\n"
    "              f'(wt1={_post_wt1:.4f} lt1={_post_lt1:.4f})')\n"
    "        print(f'Post-RL (inference-feature): wm={_inf_margin:.4f} '\n"
    "              f'(wt1={_inf_wt1:.4f} lt1={_inf_lt1:.4f}) delta={_inf_margin-_post_margin:+.4f}')\n"
    "        print(f'Forgetting: IL_baseline={_pre_margin:.4f} -> RL={_post_margin:.4f} '\n"
    "              f'delta={_post_margin - _pre_margin:+.4f}')"
)
assert old_post_eval in src23, "old_post_eval not found in Cell[23]"
src23 = src23.replace(old_post_eval, new_post_eval)

# 4. Add inference fields to _v07d2_rl_report dict
old_report_margin = (
    "        'winner_top1': float(_post_wt1), 'loser_top1': float(_post_lt1),\n"
    "        'winner_margin': float(_post_margin),"
)
new_report_margin = (
    "        'winner_top1': float(_post_wt1), 'loser_top1': float(_post_lt1),\n"
    "        'winner_margin': float(_post_margin),\n"
    "        'winner_margin_inference': float(_inf_margin) if _inf_margin is not None else None,\n"
    "        'winner_top1_inference':   float(_inf_wt1)   if _inf_wt1   is not None else None,\n"
    "        'loser_top1_inference':    float(_inf_lt1)   if _inf_lt1   is not None else None,"
)
assert old_report_margin in src23, "old_report_margin not found in Cell[23]"
src23 = src23.replace(old_report_margin, new_report_margin)

# 5. Add inference field to MAIN_HYBRID_REPORT
old_hybrid_margin = (
    "        'winner_margin': float(_post_margin),\n"
    "        'winner_top1': float(_post_wt1), 'loser_top1': float(_post_lt1),"
)
new_hybrid_margin = (
    "        'winner_margin': float(_post_margin),\n"
    "        'winner_margin_inference': float(_inf_margin) if _inf_margin is not None else None,\n"
    "        'winner_top1': float(_post_wt1), 'loser_top1': float(_post_lt1),"
)
assert old_hybrid_margin in src23, "old_hybrid_margin not found in Cell[23]"
src23 = src23.replace(old_hybrid_margin, new_hybrid_margin)

# 6. Update summary print
old_summary = (
    "    print(f'\\n=== v0-07d2 Summary ===')\n"
    "    print(f'  win_rate (diverse opp) : {_post_wr:.3f}')\n"
    "    print(f'  winner_margin          : {_post_margin:.4f}  (IL baseline: {_pre_margin:.4f})')\n"
    "    print(f'  total_time             : {_v07d2_elapsed/60:.1f}min')"
)
new_summary = (
    "    print(f'\\n=== v0-07d5 Summary ===')\n"
    "    print(f'  win_rate (diverse opp)     : {_post_wr:.3f}')\n"
    "    print(f'  winner_margin (stored)     : {_post_margin:.4f}  (IL baseline: {_pre_margin:.4f})')\n"
    "    if _inf_margin is not None:\n"
    "        print(f'  winner_margin (inference)  : {_inf_margin:.4f}  (delta vs stored: {_inf_margin-_post_margin:+.4f})')\n"
    "    print(f'  total_time                 : {_v07d2_elapsed/60:.1f}min')"
)
assert old_summary in src23, "old_summary not found in Cell[23]"
src23 = src23.replace(old_summary, new_summary)

cells[23]['source'] = src23.splitlines(keepends=True)

# ══════════════════════════════════════════════════════════════════════════════
# Clear output from all cells (fresh notebook)
# ══════════════════════════════════════════════════════════════════════════════
for c in cells:
    c['outputs'] = []
    c['execution_count'] = None

nb['metadata']['kernelspec'] = {
    'display_name': 'Python 3',
    'language': 'python',
    'name': 'python3'
}

with open(DST, 'w') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f'Written: {DST}')

# ── Sanity checks ──────────────────────────────────────────────────────────
with open(DST) as f:
    nb2 = json.load(f)
src1_check = ''.join(nb2['cells'][1]['source'])
assert 'v0_07d5_full_pipeline_new_data' in src1_check, 'EXPERIMENT_NAME not updated'
assert 'SKIP_PIPELINE      = False' in src1_check, 'SKIP_PIPELINE not False'
assert 'top200-20260624-ranking' in src1_check, 'TOP200 path not updated'
assert 'v0-07d5-remote-pc' in src1_check, 'RUN_PREFIX not updated'
assert 'episodes-2026-06-24' in src1_check, 'EPISODE_ROOT not updated'
assert 'EPISODE_SOURCE     = None' in src1_check, 'EPISODE_SOURCE not removed'
src23_check = ''.join(nb2['cells'][23]['source'])
assert 'torch.manual_seed(RANDOM_SEED)' in src23_check, 'torch.manual_seed missing'
assert '_v07d2_winner_margin_inference' in src23_check, 'inference eval fn missing'
assert 'winner_margin_inference' in src23_check, 'inference margin in report missing'
assert '_inf_wt1 = _inf_lt1 = _inf_margin = None' in src23_check, '_inf_* init missing'
print('All sanity checks passed.')
