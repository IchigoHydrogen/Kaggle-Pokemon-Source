"""Build pokemon-20260627-v0-08d14-remote-pc.ipynb from v08d2.

v08d14: game-theory derived features — computed from existing option/board data.

Add features that capture win conditions and game momentum:
  - effective_damage: op_active_hp - remain_damage_counter (clamped to >=0)
    = how much damage this option deals to the opponent's active
  - can_ko: (effective_damage >= op_active_hp) = does this attack KO opponent
  - prize_advantage: my_prizes_left - op_prizes_left (negative = we're winning)

These features encode strategic game-winning knowledge:
- If can_ko=1 → always choose this option (win condition)
- If prize_advantage < 0 → we're ahead, play safely
- If prize_advantage > 0 → we're behind, need to be aggressive

Also combines with opponent_archetype (v08d10) since these strategy shifts
are archetype-dependent.

Changes from v08d2:
  1. EXPERIMENT_NAME → v0_08d14_game_features
  2. RUN_PREFIX → pokemon-20260627-v0-08d14
  3. Cell[19]: Add derived features to work DataFrame
  4. Cell[19]: Add opponent_archetype_norm (from v08d10)
  5. Cell[19]: Update _LGBM_INJ_CODE to compute derived features at inference
"""
import json

SRC = '/kaggle/working/pokemon-20260627-v0-08d2.ipynb'
DST = '/kaggle/working/pokemon-20260627-v0-08d14.ipynb'

with open(SRC) as f:
    nb = json.load(f)

cells = nb['cells']

# ── Cell[1]: Config ───────────────────────────────────────────────────────────
src1 = ''.join(cells[1]['source'])

OLD_EXP = "EXPERIMENT_NAME = 'v0_08d2_top200_lgbm_noeval'"
NEW_EXP = "EXPERIMENT_NAME = 'v0_08d14_game_features'"
assert OLD_EXP in src1
src1 = src1.replace(OLD_EXP, NEW_EXP)

OLD_PFX = "RUN_PREFIX = 'pokemon-20260627-v0-08d2'"
NEW_PFX = "RUN_PREFIX = 'pokemon-20260627-v0-08d14'"
assert OLD_PFX in src1
src1 = src1.replace(OLD_PFX, NEW_PFX)

cells[1]['source'] = src1.splitlines(keepends=True)

# ── Cell[19]: Add derived game features + opponent_archetype ─────────────────
src19 = ''.join(cells[19]['source'])

# 1. Add opponent_archetype to categorical features
OLD_CAT = "UNKNOWN0_CATEGORICAL_FEATURES = ['context_name', 'option_type', 'area', 'in_play_area', 'option_signature']"
NEW_CAT = "UNKNOWN0_CATEGORICAL_FEATURES = ['context_name', 'option_type', 'area', 'in_play_area', 'option_signature', 'opponent_archetype_norm']  # v08d14: add archetype"
assert OLD_CAT in src19
src19 = src19.replace(OLD_CAT, NEW_CAT)

# 2. Join archetype + add derived features after work is built
OLD_WORK = (
    "        work = prepare_unknown0_training_frame(ALAKAZAM_OPTION_MODEL_DF, max_rows=max_rows)\n"
    "        # Extra features from ALAKAZAM_OPTION_MODEL_DF\n"
    "        _extra_numeric_candidates = ['supporter_played', 'stadium_played', 'energy_attached',\n"
    "                                      'my_discard_count', 'op_discard_count']"
)
NEW_WORK = (
    "        work = prepare_unknown0_training_frame(ALAKAZAM_OPTION_MODEL_DF, max_rows=max_rows)\n"
    "        # v08d14: add opponent_archetype feature\n"
    "        _v14_arch_map = {'Hop_Trevenant': 'hop_control', 'Alakazam': 'alakazam_mirror',\n"
    "                         'Mega_Lucario': 'lucario', 'Other': 'generic_control'}\n"
    "        if 'DECISION_ROWS_DF' in globals() and DECISION_ROWS_DF is not None and 'opponent_archetype' in DECISION_ROWS_DF.columns:\n"
    "            _arch_df = DECISION_ROWS_DF[['decision_id', 'opponent_archetype']].drop_duplicates('decision_id').copy()\n"
    "            _arch_df['opponent_archetype_norm'] = _arch_df['opponent_archetype'].map(_v14_arch_map).fillna('generic_control')\n"
    "            work = work.merge(_arch_df[['decision_id', 'opponent_archetype_norm']], on='decision_id', how='left')\n"
    "            work['opponent_archetype_norm'] = work['opponent_archetype_norm'].fillna('generic_control')\n"
    "        else:\n"
    "            work['opponent_archetype_norm'] = 'generic_control'\n"
    "        # v08d14: derived game-theory features\n"
    "        if 'remain_damage_counter' in work.columns and 'op_active_hp' in work.columns:\n"
    "            work['effective_damage'] = (work['op_active_hp'].fillna(0) - work['remain_damage_counter'].fillna(0)).clip(lower=0)\n"
    "            work['can_ko'] = (work['remain_damage_counter'].fillna(999) == 0).astype(float)\n"
    "            print(f'v08d14 effective_damage: mean={work[\"effective_damage\"].mean():.1f}, can_ko rate={work[\"can_ko\"].mean():.3f}')\n"
    "        if 'my_prizes_left' in work.columns and 'op_prizes_left' in work.columns:\n"
    "            work['prize_advantage'] = work['my_prizes_left'].fillna(0) - work['op_prizes_left'].fillna(0)\n"
    "            print(f'v08d14 prize_advantage: mean={work[\"prize_advantage\"].mean():.2f}')\n"
    "        # Extra features from ALAKAZAM_OPTION_MODEL_DF\n"
    "        _extra_numeric_candidates = ['supporter_played', 'stadium_played', 'energy_attached',\n"
    "                                      'my_discard_count', 'op_discard_count']"
)
assert OLD_WORK in src19
src19 = src19.replace(OLD_WORK, NEW_WORK)

# 3. Add detect_opponent_archetype + derived features to injection code
OLD_INJ_ROWS = "rows = []\\n        for i, o in enumerate(select.option):"
NEW_INJ_ROWS = (
    "_op_all = ([op_active] if op_active else []) + op_bench\\n"
    "        _op_arch, _ = detect_opponent_archetype(_op_all, float(getattr(getattr(obs.current, \\'stadium\\', None), \\'id\\', 0) or 0))\\n"
    "        rows = []\\n        for i, o in enumerate(select.option):"
)
assert OLD_INJ_ROWS in src19
src19 = src19.replace(OLD_INJ_ROWS, NEW_INJ_ROWS)

# 4. Add computed features + archetype to each row
OLD_INJ_SIG = "   \\'option_signature\\': sig,\\n            })"
NEW_INJ_SIG = (
    "   \\'option_signature\\': sig,\\n"
    "                \\'opponent_archetype_norm\\': _op_arch,\\n"
    "                \\'effective_damage\\': float(max(0, float(getattr(op_active, \\'hp\\', 0) or 0) - float(getattr(o, \\'remainDamageCounter\\', getattr(o, \\'remain_damage_counter\\', 0)) or 0))),\\n"
    "                \\'can_ko\\': float((float(getattr(o, \\'remainDamageCounter\\', getattr(o, \\'remain_damage_counter\\', 999)) or 999)) == 0),\\n"
    "                \\'prize_advantage\\': float(len([p for p in (my_ps.prize or []) if p is not None])) - float(len([p for p in (op_ps.prize or []) if p is not None])),\\n"
    "            })"
)
assert OLD_INJ_SIG in src19
src19 = src19.replace(OLD_INJ_SIG, NEW_INJ_SIG)

cells[19]['source'] = src19.splitlines(keepends=True)

# ── Cell[7]: Preload episode data ─────────────────────────────────────────────
src7 = ''.join(cells[7]['source'])

OLD_PRELOAD_POINT = 'NON_EPISODE_JSON_ROWS = []\n\nif RUN_REPLAY_MINING:'
NEW_PRELOAD_POINT = (
    "NON_EPISODE_JSON_ROWS = []\n\n"
    "# v08d14: preload from prior run cache\n"
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
    "    print(f'v08d14 preload: from {_pfx}')\n"
    "else:\n"
    "    print('v08d14 preload: no cache, mining from scratch')\n\n"
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

assert 'v0_08d14_game_features' in s1,              'EXPERIMENT_NAME'
assert "RUN_PREFIX = 'pokemon-20260627-v0-08d14'" in s1, 'RUN_PREFIX'
assert 'RUN_LOCAL_EVAL = False' in s1,              'eval disabled'
assert 'effective_damage' in s19,                   'effective_damage feature'
assert 'can_ko' in s19,                             'can_ko feature'
assert 'prize_advantage' in s19,                    'prize_advantage feature'
assert 'opponent_archetype_norm' in s19,            'archetype feature'
assert 'v08d14 preload' in s7,                      'preload logic'
full = ''.join(''.join(c['source']) for c in nb2['cells'])
assert 'lambdarank' in full, 'lambdarank kept'
print('All sanity checks passed.')
