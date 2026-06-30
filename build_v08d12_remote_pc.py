"""Build pokemon-20260627-v0-08d12-remote-pc.ipynb from v08d2.

v08d12: combine opponent_archetype (v08d10 best) + rate-weighting (v08d3 benefit).

v08d10 showed opponent_archetype gives +0.0015 improvement (new best 0.5272).
v08d3 showed rate-weighting gives +0.0005.
These are independent mechanisms — combining may compound the benefit.

Note: v08d9 showed that rate-wt + slow-LR is counterproductive (both affect
the optimizer). rate-wt + opp-arch is different — they affect DIFFERENT PARTS
of the pipeline (weights vs features).

Changes from v08d2:
  1. EXPERIMENT_NAME → v0_08d12_opp_arch_rate_wt
  2. RUN_PREFIX → pokemon-20260627-v0-08d12
  3. Cell[19]: rate-weighted formula (v08d3)
  4. Cell[19]: opponent_archetype_norm feature (v08d10)
  5. Cell[19]: _LGBM_INJ_CODE: detect_opponent_archetype call (v08d10)
"""
import json

SRC = '/kaggle/working/pokemon-20260627-v0-08d2.ipynb'
DST = '/kaggle/working/pokemon-20260627-v0-08d12.ipynb'

with open(SRC) as f:
    nb = json.load(f)

cells = nb['cells']

# ── Cell[1]: Config ───────────────────────────────────────────────────────────
src1 = ''.join(cells[1]['source'])

OLD_EXP = "EXPERIMENT_NAME = 'v0_08d2_top200_lgbm_noeval'"
NEW_EXP = "EXPERIMENT_NAME = 'v0_08d12_opp_arch_rate_wt'"
assert OLD_EXP in src1
src1 = src1.replace(OLD_EXP, NEW_EXP)

OLD_PFX = "RUN_PREFIX = 'pokemon-20260627-v0-08d2'"
NEW_PFX = "RUN_PREFIX = 'pokemon-20260627-v0-08d12'"
assert OLD_PFX in src1
src1 = src1.replace(OLD_PFX, NEW_PFX)

cells[1]['source'] = src1.splitlines(keepends=True)

# ── Cell[19]: rate-weighted formula (v08d3) + opp_arch (v08d10) ──────────────
src19 = ''.join(cells[19]['source'])

# 1. Rate-weighted training (from v08d3)
OLD_WT = (
    "            # Winner-weighted training: winner decisions get 2x weight\n"
    "            if 'won' in train_df.columns:\n"
    "                _winner_weight = 4.0\n"
    "                train_df = train_df.copy()\n"
    "                train_df['_win_wt'] = train_df['won'].fillna(0).astype(float) * (_winner_weight - 1.0) + 1.0\n"
    "                _n_winner = int((train_df['won'] == True).sum())\n"
    "                print(f'Winner-weighted training: {_n_winner}/{len(train_df)} winner rows (weight={_winner_weight}x)')"
)
NEW_WT = (
    "            # v08d12: rate-weighted training (v08d3) — higher-rated player decisions get more weight\n"
    "            if 'won' in train_df.columns:\n"
    "                _winner_weight = 4.0\n"
    "                train_df = train_df.copy()\n"
    "                _base_wt = train_df['won'].fillna(0).astype(float) * (_winner_weight - 1.0) + 1.0\n"
    "                if 'rate' in train_df.columns:\n"
    "                    _mean_rate = train_df['rate'].fillna(train_df['rate'].mean()).mean()\n"
    "                    _rate_factor = train_df['rate'].fillna(_mean_rate) / _mean_rate\n"
    "                    train_df['_win_wt'] = _base_wt * _rate_factor\n"
    "                    print(f'Rate-weighted training: mean_rate={_mean_rate:.1f}, rate_factor range=[{_rate_factor.min():.2f}, {_rate_factor.max():.2f}]')\n"
    "                else:\n"
    "                    train_df['_win_wt'] = _base_wt\n"
    "                    print('rate column not found, using winner_weight only')\n"
    "                _n_winner = int((train_df['won'] == True).sum())\n"
    "                print(f'Winner-weighted training: {_n_winner}/{len(train_df)} winner rows (base weight={_winner_weight}x, rate-scaled)')"
)
assert OLD_WT in src19, 'winner_weight formula not found in Cell[19]'
src19 = src19.replace(OLD_WT, NEW_WT)

# 2. Add opponent_archetype_norm to categorical features (from v08d10)
OLD_CAT = "UNKNOWN0_CATEGORICAL_FEATURES = ['context_name', 'option_type', 'area', 'in_play_area', 'option_signature']"
NEW_CAT = "UNKNOWN0_CATEGORICAL_FEATURES = ['context_name', 'option_type', 'area', 'in_play_area', 'option_signature', 'opponent_archetype_norm']  # v08d12: add archetype"
assert OLD_CAT in src19
src19 = src19.replace(OLD_CAT, NEW_CAT)

# 3. Join opponent_archetype from DECISION_ROWS_DF (from v08d10)
OLD_WORK = (
    "        work = prepare_unknown0_training_frame(ALAKAZAM_OPTION_MODEL_DF, max_rows=max_rows)\n"
    "        # Extra features from ALAKAZAM_OPTION_MODEL_DF\n"
    "        _extra_numeric_candidates = ['supporter_played', 'stadium_played', 'energy_attached',\n"
    "                                      'my_discard_count', 'op_discard_count']"
)
NEW_WORK = (
    "        work = prepare_unknown0_training_frame(ALAKAZAM_OPTION_MODEL_DF, max_rows=max_rows)\n"
    "        # v08d12: join opponent_archetype from DECISION_ROWS_DF (same as v08d10)\n"
    "        _v12_arch_map = {'Hop_Trevenant': 'hop_control', 'Alakazam': 'alakazam_mirror',\n"
    "                         'Mega_Lucario': 'lucario', 'Other': 'generic_control'}\n"
    "        if 'DECISION_ROWS_DF' in globals() and DECISION_ROWS_DF is not None and 'opponent_archetype' in DECISION_ROWS_DF.columns:\n"
    "            _arch_df = DECISION_ROWS_DF[['decision_id', 'opponent_archetype']].drop_duplicates('decision_id').copy()\n"
    "            _arch_df['opponent_archetype_norm'] = _arch_df['opponent_archetype'].map(_v12_arch_map).fillna('generic_control')\n"
    "            work = work.merge(_arch_df[['decision_id', 'opponent_archetype_norm']], on='decision_id', how='left')\n"
    "            work['opponent_archetype_norm'] = work['opponent_archetype_norm'].fillna('generic_control')\n"
    "            print(f'v08d12 archetype distribution: {work[\"opponent_archetype_norm\"].value_counts().to_dict()}')\n"
    "        else:\n"
    "            work['opponent_archetype_norm'] = 'generic_control'\n"
    "        # Extra features from ALAKAZAM_OPTION_MODEL_DF\n"
    "        _extra_numeric_candidates = ['supporter_played', 'stadium_played', 'energy_attached',\n"
    "                                      'my_discard_count', 'op_discard_count']"
)
assert OLD_WORK in src19
src19 = src19.replace(OLD_WORK, NEW_WORK)

# 4. Add detect_opponent_archetype to injection code (from v08d10)
OLD_INJ_ROWS = "rows = []\\n        for i, o in enumerate(select.option):"
NEW_INJ_ROWS = (
    "_op_all = ([op_active] if op_active else []) + op_bench\\n"
    "        _op_arch, _ = detect_opponent_archetype(_op_all, float(getattr(getattr(obs.current, \\'stadium\\', None), \\'id\\', 0) or 0))\\n"
    "        rows = []\\n        for i, o in enumerate(select.option):"
)
assert OLD_INJ_ROWS in src19
src19 = src19.replace(OLD_INJ_ROWS, NEW_INJ_ROWS)

OLD_INJ_SIG = "   \\'option_signature\\': sig,\\n            })"
NEW_INJ_SIG = "   \\'option_signature\\': sig,\\n                \\'opponent_archetype_norm\\': _op_arch,\\n            })"
assert OLD_INJ_SIG in src19
src19 = src19.replace(OLD_INJ_SIG, NEW_INJ_SIG)

cells[19]['source'] = src19.splitlines(keepends=True)

# ── Cell[7]: Preload episode data ─────────────────────────────────────────────
src7 = ''.join(cells[7]['source'])

OLD_PRELOAD_POINT = 'NON_EPISODE_JSON_ROWS = []\n\nif RUN_REPLAY_MINING:'
NEW_PRELOAD_POINT = (
    "NON_EPISODE_JSON_ROWS = []\n\n"
    "# v08d12: preload from prior run cache\n"
    "_PRELOAD_CANDIDATES = [\n"
    "    Path('/kaggle/working/pokemon-20260627-v0-08d2/pokemon-20260627-v0-08d2'),\n"
    "    Path('/kaggle/working/pokemon-20260627-v0-08d1/pokemon-20260627-v0-08d1'),\n"
    "]\n"
    "_PRELOAD_PFX = None\n"
    "for _cand in _PRELOAD_CANDIDATES:\n"
    "    if (_cand.parent.exists() and\n"
    "            (_cand.parent / f'{_cand.name}-decision_rows.parquet').exists()):\n"
    "        _PRELOAD_PFX = str(_cand)\n"
    "        break\n"
    "_preload_ok = _PRELOAD_PFX is not None\n"
    "if _preload_ok:\n"
    "    _pfx = _PRELOAD_PFX\n"
    "    DECISION_ROWS_DF  = pd.read_parquet(f'{_pfx}-decision_rows.parquet')\n"
    "    OPTION_ROWS_DF    = pd.read_parquet(f'{_pfx}-option_rows.parquet')\n"
    "    EPISODE_INDEX_DF  = pd.read_parquet(f'{_pfx}-episode_index.parquet')\n"
    "    EPISODE_SPLIT_DF  = pd.read_parquet(f'{_pfx}-episode_split.parquet')\n"
    "    DECKLISTS_DF      = pd.read_parquet(f'{_pfx}-decklists.parquet')\n"
    "    STATE_SUMMARY_DF  = pd.read_parquet(f'{_pfx}-state_summary.parquet')\n"
    "    RUN_REPLAY_MINING = False\n"
    "    print(f'v08d12 preload: {len(DECISION_ROWS_DF)} decisions from {_pfx}')\n"
    "else:\n"
    "    print('v08d12 preload: no cache, mining from scratch')\n\n"
    "if RUN_REPLAY_MINING:"
)
assert OLD_PRELOAD_POINT in src7
src7 = src7.replace(OLD_PRELOAD_POINT, NEW_PRELOAD_POINT)

cells[7]['source'] = src7.splitlines(keepends=True)

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
s1 = ''.join(nb2['cells'][1]['source'])
s19 = ''.join(nb2['cells'][19]['source'])
s7 = ''.join(nb2['cells'][7]['source'])

assert 'v0_08d12_opp_arch_rate_wt' in s1,           'EXPERIMENT_NAME'
assert "RUN_PREFIX = 'pokemon-20260627-v0-08d12'" in s1, 'RUN_PREFIX'
assert 'RUN_LOCAL_EVAL = False' in s1,               'eval disabled'
assert 'rate_factor' in s19,                         'rate_factor in training'
assert 'opponent_archetype_norm' in s19,             'archetype feature'
assert '_op_arch' in s19,                            'archetype inference'
assert 'v08d12 preload' in s7,                       'preload logic'
full = ''.join(''.join(c['source']) for c in nb2['cells'])
assert 'lambdarank' in full, 'lambdarank kept'
print('All sanity checks passed.')
