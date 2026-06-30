"""Build pokemon-20260627-v0-08d31-remote-pc.ipynb from v08d28.

v08d31: Add my_active_id, my_active_hp, op_active_hp, powerful_hand_can_ko_active.

DISCOVERY: alakazam_option_model_df.parquet contains these columns that are
NOT currently in _extra_numeric_candidates:
  - my_active_id: which of MY pokemon is active (mirror of op_active_id=72-80k!)
  - my_active_hp: HP of my active pokemon
  - op_active_hp: HP of opponent's active pokemon (NOT currently in model!)
  - powerful_hand_can_ko_active: binary, can I KO opponent's active pokemon?

HYPOTHESIS: These are the missing "battle state" features. The model knows
op_active_id (72-80k importance) but NOT my_active_id. This asymmetry means
the model can't reason about MY pokemon's matchup against the opponent.

Expected impact:
  - my_active_id: ~similar to op_active_id (~70k importance potential)
  - my_active_hp + op_active_hp: HP ratio determines urgency of attacking
  - powerful_hand_can_ko_active: direct KO opportunity indicator

Based on v08d28 (includes position_winrate, +0.0008).

Changes from v08d28:
  1. EXPERIMENT_NAME → v0_08d31_active_feats
  2. RUN_PREFIX → pokemon-20260627-v0-08d31
  3. Cell[7]: Preload from v08d19 (same split)
  4. Cell[19]: Add new features to _extra_numeric_candidates
"""
import json

SRC = '/kaggle/working/pokemon-20260627-v0-08d28.ipynb'
DST = '/kaggle/working/pokemon-20260627-v0-08d31.ipynb'

with open(SRC) as f:
    nb = json.load(f)

cells = nb['cells']

# ── Cell[1]: Config ───────────────────────────────────────────────────────────
src1 = ''.join(cells[1]['source'])

OLD_EXP = "EXPERIMENT_NAME = 'v0_08d28_winrate_feat'"
NEW_EXP = "EXPERIMENT_NAME = 'v0_08d31_active_feats'"
assert OLD_EXP in src1
src1 = src1.replace(OLD_EXP, NEW_EXP)

OLD_PFX = "RUN_PREFIX = 'pokemon-20260627-v0-08d28'"
NEW_PFX = "RUN_PREFIX = 'pokemon-20260627-v0-08d31'"
assert OLD_PFX in src1
src1 = src1.replace(OLD_PFX, NEW_PFX)

cells[1]['source'] = src1.splitlines(keepends=True)

# ── Cell[7]: Preload ──────────────────────────────────────────────────────────
src7 = ''.join(cells[7]['source'])

OLD_PRELOAD = "# v08d28: preload from v08d19 (same split)\n"
NEW_PRELOAD = "# v08d31: preload from v08d19 (same split)\n"
assert OLD_PRELOAD in src7
src7 = src7.replace(OLD_PRELOAD, NEW_PRELOAD)

OLD_PRINT = "    print(f'v08d28 preload: {len(DECISION_ROWS_DF)} decisions from {_pfx}')\n"
NEW_PRINT = "    print(f'v08d31 preload: {len(DECISION_ROWS_DF)} decisions from {_pfx}')\n"
assert OLD_PRINT in src7
src7 = src7.replace(OLD_PRINT, NEW_PRINT)

OLD_FAIL = "    print('v08d28 preload: no cache found, mining from scratch')\n"
NEW_FAIL = "    print('v08d31 preload: no cache found, mining from scratch')\n"
assert OLD_FAIL in src7
src7 = src7.replace(OLD_FAIL, NEW_FAIL)

cells[7]['source'] = src7.splitlines(keepends=True)

# ── Cell[19]: Add active features to _extra_numeric_candidates ────────────────
src19 = ''.join(cells[19]['source'])

OLD_CANDS = (
    "        _extra_numeric_candidates = ['supporter_played', 'stadium_played', 'energy_attached',\n"
    "                                      'my_discard_count', 'op_discard_count', 'prize_gap', 'steps_since_op',\n"
    "                                      'position_winrate']\n"
)
NEW_CANDS = (
    "        # v08d31: add battle-state features (all exist in ALAKAZAM_OPTION_MODEL_DF)\n"
    "        # my_active_id: mirror of op_active_id (72-80k importance); op_active_hp: missing asymmetry\n"
    "        _extra_numeric_candidates = ['supporter_played', 'stadium_played', 'energy_attached',\n"
    "                                      'my_discard_count', 'op_discard_count', 'prize_gap', 'steps_since_op',\n"
    "                                      'position_winrate',\n"
    "                                      'my_active_id', 'my_active_hp', 'op_active_hp',\n"
    "                                      'powerful_hand_can_ko_active']\n"
)
assert OLD_CANDS in src19, '_extra_numeric_candidates not found in Cell[19]'
src19 = src19.replace(OLD_CANDS, NEW_CANDS)

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
s7  = ''.join(nb2['cells'][7]['source'])
s19 = ''.join(nb2['cells'][19]['source'])

assert 'v0_08d31_active_feats' in s1
assert "RUN_PREFIX = 'pokemon-20260627-v0-08d31'" in s1
assert 'v08d19' in s7
assert 'my_active_id' in s19
assert 'my_active_hp' in s19
assert 'op_active_hp' in s19
assert 'powerful_hand_can_ko_active' in s19
assert 'position_winrate' in s19, 'position_winrate must be kept from v08d28'
assert 'op_last_context' in s19
assert '_winner_weight = 4.0' in s19
full = ''.join(''.join(c['source']) for c in nb2['cells'])
assert 'lambdarank' in full
print('All sanity checks passed.')
