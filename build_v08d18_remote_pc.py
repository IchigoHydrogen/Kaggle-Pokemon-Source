"""Build pokemon-20260627-v0-08d18-remote-pc.ipynb from v08d10.

v08d18: prize_gap feature — fix prizes_left mining bug + add prize_gap to LGBM.

The Bug (found via data exploration):
  count_zone(my, 'prize', 'prizeCount') counts non-None items in prize list.
  Prize cards are face-down → stored as [None, None, None] → count = 0 always.
  Fix: len(prize_list) counts ALL items (None = face-down card still occupies slot).

New Feature:
  prize_gap = my_prizes_left - op_prizes_left
    Negative = we're ahead (opponent needs to take more prizes to win)
    Positive = we're behind
  Also add my_prizes_left and op_prizes_left as direct features.

This is the single most important missing game-state signal: who's winning.
Requires full re-mining (no preload) since the fix is in mining code.

Changes from v08d10:
  1. EXPERIMENT_NAME → v0_08d18_prize_gap
  2. RUN_PREFIX → pokemon-20260627-v0-08d18
  3. Cell[6]: Fix my_prizes_left / op_prizes_left in state_summary_from_obs
  4. Cell[7]: Remove preload (force fresh mining)
  5. Cell[19]: Add prize_gap to work DataFrame after opp_arch join
  6. Cell[19]: Add prize_gap to LGBM injection code
"""
import json

SRC = '/kaggle/working/pokemon-20260627-v0-08d10.ipynb'
DST = '/kaggle/working/pokemon-20260627-v0-08d18.ipynb'

with open(SRC) as f:
    nb = json.load(f)

cells = nb['cells']

# ── Cell[1]: Config ───────────────────────────────────────────────────────────
src1 = ''.join(cells[1]['source'])

OLD_EXP = "EXPERIMENT_NAME = 'v0_08d10_opp_arch'"
NEW_EXP = "EXPERIMENT_NAME = 'v0_08d18_prize_gap'"
assert OLD_EXP in src1, f'EXPERIMENT_NAME not found: {OLD_EXP!r}'
src1 = src1.replace(OLD_EXP, NEW_EXP)

OLD_PFX = "RUN_PREFIX = 'pokemon-20260627-v0-08d10'"
NEW_PFX = "RUN_PREFIX = 'pokemon-20260627-v0-08d18'"
assert OLD_PFX in src1, f'RUN_PREFIX not found: {OLD_PFX!r}'
src1 = src1.replace(OLD_PFX, NEW_PFX)

cells[1]['source'] = src1.splitlines(keepends=True)

# ── Cell[6]: Fix prizes_left mining bug ───────────────────────────────────────
src6 = ''.join(cells[6]['source'])

OLD_PRIZES = (
    "        'my_prizes_left': count_zone(my, 'prize', 'prizeCount'),\n"
    "        'op_prizes_left': count_zone(op, 'prize', 'prizeCount'),"
)
NEW_PRIZES = (
    "        # v08d18: fix prizes bug — prize list is [None,None,...] (face-down cards)\n"
    "        # count_zone filtered None → always 0. len() counts all slots correctly.\n"
    "        'my_prizes_left': len((my.get('prize') or [])),\n"
    "        'op_prizes_left': len((op.get('prize') or [])),"
)
assert OLD_PRIZES in src6, 'prizes_left lines not found in Cell[6]'
src6 = src6.replace(OLD_PRIZES, NEW_PRIZES)

cells[6]['source'] = src6.splitlines(keepends=True)

# ── Cell[7]: Remove preload (force fresh mining for prize fix) ────────────────
src7 = ''.join(cells[7]['source'])

# Remove the entire preload block, keeping only the RUN_REPLAY_MINING gate
OLD_PRELOAD_BLOCK = (
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
    "    print(f'v08d10 preload: {len(DECISION_ROWS_DF)} decisions, {len(OPTION_ROWS_DF)} options from {_pfx}')\n"
    "else:\n"
    "    print('v08d10 preload: no cache found, running episode mining from scratch')\n\n"
    "if RUN_REPLAY_MINING:"
)
NEW_NO_PRELOAD = (
    "# v08d18: NO preload — prizes_left bug fix requires fresh mining\n"
    "# Preloaded parquets have prizes_left=0 (old bug). Must re-mine.\n"
    "print('v08d18: forcing fresh episode mining (prizes_left fix requires it)')\n\n"
    "if RUN_REPLAY_MINING:"
)
assert OLD_PRELOAD_BLOCK in src7, 'preload block not found in Cell[7]'
src7 = src7.replace(OLD_PRELOAD_BLOCK, NEW_NO_PRELOAD)

cells[7]['source'] = src7.splitlines(keepends=True)

# ── Cell[19]: Add prize_gap feature ──────────────────────────────────────────
src19 = ''.join(cells[19]['source'])

# 1. Add prize features to work DataFrame (after opp_arch join from v08d10)
OLD_EXTRA = (
    "        # Extra features from ALAKAZAM_OPTION_MODEL_DF\n"
    "        _extra_numeric_candidates = ['supporter_played', 'stadium_played', 'energy_attached',\n"
    "                                      'my_discard_count', 'op_discard_count']"
)
NEW_EXTRA = (
    "        # v08d18: prize_gap feature (my_prizes_left and op_prizes_left now correct after bug fix)\n"
    "        if 'my_prizes_left' in work.columns and 'op_prizes_left' in work.columns:\n"
    "            work['prize_gap'] = (work['my_prizes_left'].fillna(0) - work['op_prizes_left'].fillna(0))\n"
    "            _prize_dist = work['prize_gap'].value_counts().sort_index().to_dict()\n"
    "            print(f'v08d18 prize_gap dist: {_prize_dist}')\n"
    "            print(f'v08d18 my_prizes_left: mean={work[\"my_prizes_left\"].mean():.2f}')\n"
    "            print(f'v08d18 op_prizes_left: mean={work[\"op_prizes_left\"].mean():.2f}')\n"
    "        else:\n"
    "            work['prize_gap'] = 0.0\n"
    "            print('v08d18 WARNING: prizes_left columns missing from work')\n"
    "        # Extra features from ALAKAZAM_OPTION_MODEL_DF\n"
    "        _extra_numeric_candidates = ['supporter_played', 'stadium_played', 'energy_attached',\n"
    "                                      'my_discard_count', 'op_discard_count', 'prize_gap']"
)
assert OLD_EXTRA in src19, 'extra_numeric_candidates not found in Cell[19]'
src19 = src19.replace(OLD_EXTRA, NEW_EXTRA)

# 2. Add prize_gap to injection code (before rows = [])
# v08d10 already has: _op_arch computed before rows = []
OLD_INJ_ROWS = (
    "_op_all = ([op_active] if op_active else []) + op_bench\\n"
    "        _op_arch, _ = detect_opponent_archetype(_op_all, float(getattr(getattr(obs.current, \\'stadium\\', None), \\'id\\', 0) or 0))\\n"
    "        rows = []\\n        for i, o in enumerate(select.option):"
)
NEW_INJ_ROWS = (
    "_op_all = ([op_active] if op_active else []) + op_bench\\n"
    "        _op_arch, _ = detect_opponent_archetype(_op_all, float(getattr(getattr(obs.current, \\'stadium\\', None), \\'id\\', 0) or 0))\\n"
    "        # v08d18: prize_gap — len(prize list) counts face-down prize cards correctly\\n"
    "        _my_prizes_left = float(len(list(getattr(my_ps, \\'prize\\', None) or [])))\\n"
    "        _op_prizes_left = float(len(list(getattr(op_ps, \\'prize\\', None) or [])))\\n"
    "        _prize_gap = _my_prizes_left - _op_prizes_left\\n"
    "        rows = []\\n        for i, o in enumerate(select.option):"
)
assert OLD_INJ_ROWS in src19, 'injection rows= line not found in Cell[19]'
src19 = src19.replace(OLD_INJ_ROWS, NEW_INJ_ROWS)

# 3. Add prize_gap to each row dict (after opp_arch)
OLD_INJ_END = (
    "                \\'opponent_archetype_norm\\': _op_arch,\\n"
    "            })"
)
NEW_INJ_END = (
    "                \\'opponent_archetype_norm\\': _op_arch,\\n"
    "                \\'prize_gap\\': _prize_gap,\\n"
    "            })"
)
assert OLD_INJ_END in src19, 'injection row dict end not found in Cell[19]'
src19 = src19.replace(OLD_INJ_END, NEW_INJ_END)

cells[19]['source'] = src19.splitlines(keepends=True)

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
s6  = ''.join(nb2['cells'][6]['source'])
s7  = ''.join(nb2['cells'][7]['source'])
s19 = ''.join(nb2['cells'][19]['source'])

assert 'v0_08d18_prize_gap' in s1,                     'EXPERIMENT_NAME'
assert "RUN_PREFIX = 'pokemon-20260627-v0-08d18'" in s1, 'RUN_PREFIX'
assert 'RUN_LOCAL_EVAL = False' in s1,                  'eval disabled'
assert "len((my.get('prize') or []))" in s6,            'prizes_left fix in Cell[6]'
assert "len((op.get('prize') or []))" in s6,            'op prizes_left fix in Cell[6]'
assert '_PRELOAD_CANDIDATES' not in s7,                  'preload removed'
assert 'forcing fresh episode mining' in s7,             'no-preload message'
assert 'prize_gap' in s19,                              'prize_gap feature in Cell[19]'
assert '_prize_gap' in s19,                             'prize_gap in injection code'
assert '_my_prizes_left' in s19,                        'my_prizes_left in injection'
assert 'opponent_archetype_norm' in s19,                'opp_arch kept'
full = ''.join(''.join(c['source']) for c in nb2['cells'])
assert 'lambdarank' in full, 'lambdarank kept'
print('All sanity checks passed.')
