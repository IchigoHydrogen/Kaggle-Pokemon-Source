"""Build v07d3 notebook from v07d2.

Changes:
1. Cell[1]: EXPERIMENT_NAME, RUN_PREFIX bump
2. Cell[23]: accumulate episodes per iter + torch.save after loop
3. Cell[25]: fix promotion criterion (winner_margin > 0.030) + enrich _promo dict
"""
import json, re, sys
from pathlib import Path

SRC  = Path('/kaggle/working/pokemon-20260625-v0-07d2.ipynb')
DST  = Path('/kaggle/working/pokemon-20260625-v0-07d3.ipynb')

nb = json.loads(SRC.read_text())

# ── Cell[1]: bump identifiers ────────────────────────────────────────────────
src1 = ''.join(nb['cells'][1]['source'])
src1 = src1.replace(
    "EXPERIMENT_NAME = 'v0_07d2_hybrid_il_rl'",
    "EXPERIMENT_NAME = 'v0_07d3_episode_separation'",
)
src1 = src1.replace(
    "RUN_PREFIX = os.environ.get('V05_RUN_PREFIX', 'pokemon-20260625-v0-07d2')",
    "RUN_PREFIX = os.environ.get('V05_RUN_PREFIX', 'pokemon-20260625-v0-07d3')",
)
nb['cells'][1]['source'] = [src1]
print('Cell[1] patched')

# ── Cell[23]: episode buffer accumulation + save ─────────────────────────────
src23 = ''.join(nb['cells'][23]['source'])

# 1. Init episode buffer before the main RL loop
OLD_HIST_INIT = '    _t_rl_start    = _v07d2_time.time()\n    _v07d2_history = []'
NEW_HIST_INIT = (
    '    _t_rl_start    = _v07d2_time.time()\n'
    '    _v07d2_history = []\n'
    '\n'
    '    # ── Episode buffer (accumulated across all iters for cross-version reuse) ──\n'
    '    _ep_feats, _ep_nopts, _ep_acts, _ep_lps = [], [], [], []\n'
    '    _ep_vals, _ep_srews, _ep_advs, _ep_rets, _ep_iters = [], [], [], [], []'
)
assert OLD_HIST_INIT in src23, 'PATCH1 anchor not found'
src23 = src23.replace(OLD_HIST_INIT, NEW_HIST_INIT, 1)

# 2. Accumulate each iter's rollout into the buffer (after GAE computation)
OLD_GAE_LINE = '        _adv, _ret = _v07d2_gae(_ro)\n        _m         = _v07d2_update(_ro, _adv, _ret, _v07d2_device)'
NEW_GAE_LINE = (
    '        _adv, _ret = _v07d2_gae(_ro)\n'
    '        _ep_feats.extend(_ro[\'feats\']); _ep_nopts.extend(_ro[\'n_opts\'])\n'
    '        _ep_acts.extend(_ro[\'actions\']); _ep_lps.extend(_ro[\'log_probs\'])\n'
    '        _ep_vals.extend(_ro[\'values\']); _ep_srews.extend(_ro[\'step_rewards\'])\n'
    '        _ep_advs.extend(_adv.tolist()); _ep_rets.extend(_ret.tolist())\n'
    '        _ep_iters.extend([_rl_iter] * len(_ro[\'actions\']))\n'
    '        _m         = _v07d2_update(_ro, _adv, _ret, _v07d2_device)'
)
assert OLD_GAE_LINE in src23, 'PATCH2 anchor not found'
src23 = src23.replace(OLD_GAE_LINE, NEW_GAE_LINE, 1)

# 3. Save episode buffer after post-RL evaluation (before model save)
OLD_MODEL_SAVE = '    _v07d2_model_path = OUTPUT_DIR / (RUN_PREFIX + \'-main_option_scorer.pt\')'
NEW_MODEL_SAVE = (
    '    # ── Save episode buffer for cross-version reuse ───────────────────────────\n'
    '    _ep_path = OUTPUT_DIR / (RUN_PREFIX + \'-rl_episodes.pt\')\n'
    '    torch.save({\n'
    '        \'feats\': _ep_feats, \'n_opts\': _ep_nopts, \'actions\': _ep_acts,\n'
    '        \'log_probs_beh\': _ep_lps, \'values_beh\': _ep_vals,\n'
    '        \'step_rewards\': _ep_srews, \'adv\': _ep_advs, \'ret\': _ep_rets,\n'
    '        \'iter_idx\': _ep_iters,\n'
    '        \'meta\': {\n'
    '            \'behavioral_version\': RUN_PREFIX,\n'
    '            \'n_iters\': _V07D2_N_ITERS, \'games_per_iter\': _V07D2_GAMES_PER_ITER,\n'
    '            \'total_decisions\': len(_ep_acts),\n'
    '            \'prize_step_coef\': _V07D2_PRIZE_STEP_COEF,\n'
    '            \'lambda_rl\': _V07D2_LAMBDA_RL, \'lambda_il\': _V07D2_LAMBDA_IL,\n'
    '            \'opponent_mix\': \'35pct_rule/30pct_IL/35pct_self\',\n'
    '        },\n'
    '    }, str(_ep_path))\n'
    '    ARTIFACT_PATHS[\'v07d3_rl_episodes\'] = str(_ep_path)\n'
    '    print(f\'Episode buffer saved: {len(_ep_acts)} decisions → {_ep_path.name}\')\n'
    '\n'
    '    _v07d2_model_path = OUTPUT_DIR / (RUN_PREFIX + \'-main_option_scorer.pt\')'
)
assert OLD_MODEL_SAVE in src23, 'PATCH3 anchor not found'
src23 = src23.replace(OLD_MODEL_SAVE, NEW_MODEL_SAVE, 1)

nb['cells'][23]['source'] = [src23]
print('Cell[23] patched (3 hunks)')

# ── Cell[25]: fix promotion criterion + enrich _promo dict ──────────────────
src25 = ''.join(nb['cells'][25]['source'])

# 4. Replace promotion criterion block
OLD_CRITERION = (
    '    _rl_win_rate   = float(MAIN_HYBRID_REPORT.get(\'win_rate_vs_rule_agent\', 0.0))\n'
    '    _winner_margin = float(MAIN_HYBRID_REPORT.get(\'winner_margin\', 0.0))\n'
    '    if _all_gates and _rl_win_rate > 0.50:\n'
    '        _promotion_decision = \'rl_promote\'\n'
    '    elif _all_gates and _rl_win_rate > 0.45:\n'
    '        _promotion_decision = \'needs_followup\'\n'
    '    else:\n'
    '        _promotion_decision = \'reject\''
)
NEW_CRITERION = (
    '    _winner_margin  = float(MAIN_HYBRID_REPORT.get(\'winner_margin\', 0.0))\n'
    '    _wm_threshold   = 0.030\n'
    '    _quality_ok     = bool(MAIN_HYBRID_REPORT.get(\'quality_ok\', False))\n'
    '    if _all_gates and _quality_ok and _winner_margin > _wm_threshold:\n'
    '        _promotion_decision = \'learning_promote\'\n'
    '    elif _all_gates and _winner_margin > 0.0:\n'
    '        _promotion_decision = \'needs_followup\'\n'
    '    else:\n'
    '        _promotion_decision = \'reject\''
)
assert OLD_CRITERION in src25, 'PATCH4 anchor not found'
src25 = src25.replace(OLD_CRITERION, NEW_CRITERION, 1)

# 5. Enrich _promo dict: insert new fields after 'decision' line
OLD_PROMO_HEAD = (
    "        'version': RUN_PREFIX,\n"
    "        'decision': _promotion_decision,\n"
    "        'promotion_type': _promotion_decision,"
)
NEW_PROMO_HEAD = (
    "        'version': RUN_PREFIX,\n"
    "        'decision': _promotion_decision,\n"
    "        'promotion_type': _promotion_decision,\n"
    "        'winner_margin': float(_winner_margin),\n"
    "        'winner_margin_threshold': float(_wm_threshold),\n"
    "        'quality_ok': bool(_quality_ok),\n"
    "        'winner_top1': float(MAIN_HYBRID_REPORT.get('winner_top1', 0.0)),\n"
    "        'loser_top1': float(MAIN_HYBRID_REPORT.get('loser_top1', 0.0)),\n"
    "        'il_baseline_winner_margin': float(MAIN_LEARNING_REPORT.get('il_baseline_winner_margin', 0.0)),\n"
    "        'il_source': str(MAIN_LEARNING_REPORT.get('il_source', '')),\n"
    "        'post_rl_win_rate': float(MAIN_LEARNING_REPORT.get('post_rl_win_rate', 0.0)),\n"
    "        'pre_rl_win_rate': float(MAIN_LEARNING_REPORT.get('pre_rl_win_rate', 0.0)),\n"
    "        'episode_source': 'fresh (behavioral policy: ' + RUN_PREFIX + ')',"
)
assert OLD_PROMO_HEAD in src25, 'PATCH5 anchor not found'
src25 = src25.replace(OLD_PROMO_HEAD, NEW_PROMO_HEAD, 1)

nb['cells'][25]['source'] = [src25]
print('Cell[25] patched (2 hunks)')

# ── Write output ─────────────────────────────────────────────────────────────
DST.write_text(json.dumps(nb, ensure_ascii=False, indent=1))
print(f'\nWrote {DST}')
print('Done.')
