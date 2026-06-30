"""Build pokemon-20260627-v0-08d32-remote-pc.ipynb from v08d28.

v08d32: Evolution state + opponent resources + damage estimate features.

LESSON FROM v08d31: features that are near-constant in UNKNOWN_0 context
(my_active_id=66.8% Alakazam, powerful_hand_can_ko_active=82.6% True)
get zero LGBM importance even when available in work.

THESE FEATURES HAVE GOOD VARIANCE:
  - my_alakazam_count: top=1 (35.3%), mean=1.31 — EVOLUTION STATE (setup vs attack phase)
  - my_kadabra_count: top=0 (57.8%), mean=0.54 — mid-evolution
  - my_abra_count: top=0 (53.0%), mean=0.71 — early-evolution
  - op_bench_count: top=5 (37.7%), mean=3.57 — opponent bench vulnerability
  - op_hand_count: top=6 (15.7%), mean=7.49 — VERY well distributed, opponent resources
  - powerful_hand_damage_est: top=260 (8.3%), mean=274.94 — EXCELLENT distribution
  - turn_action_count: top=1 (12.9%), mean=6.85 — actions taken this turn
  - my_bench_count: top=4 (37.0%), mean=3.64 — my bench size

KEY INSIGHT: my_alakazam_count captures the game PHASE (setup=0, attacking=1+).
In UNKNOWN_0 context, having 0 Alakazam means we're still evolving — completely
different strategic decisions vs having Alakazam in play.

powerful_hand_damage_est (nunique=32) is far more informative than the binary
powerful_hand_can_ko_active (which was 82.6% True).

No inference proxies needed — all features are directly in ALAKAZAM_OPTION_MODEL_DF
(same code path at train and inference time).

Based on v08d28 (includes position_winrate, +0.0008).

Changes from v08d28:
  1. EXPERIMENT_NAME → v0_08d32_evo_state
  2. RUN_PREFIX → pokemon-20260627-v0-08d32
  3. Cell[7]: Preload from v08d19 (same split)
  4. Cell[19]: Add 8 new features to _extra_numeric_candidates
"""
import json

SRC = '/kaggle/working/pokemon-20260627-v0-08d28.ipynb'
DST = '/kaggle/working/pokemon-20260627-v0-08d32.ipynb'

with open(SRC) as f:
    nb = json.load(f)

cells = nb['cells']

# ── Cell[1]: Config ───────────────────────────────────────────────────────────
src1 = ''.join(cells[1]['source'])

OLD_EXP = "EXPERIMENT_NAME = 'v0_08d28_winrate_feat'"
NEW_EXP = "EXPERIMENT_NAME = 'v0_08d32_evo_state'"
assert OLD_EXP in src1
src1 = src1.replace(OLD_EXP, NEW_EXP)

OLD_PFX = "RUN_PREFIX = 'pokemon-20260627-v0-08d28'"
NEW_PFX = "RUN_PREFIX = 'pokemon-20260627-v0-08d32'"
assert OLD_PFX in src1
src1 = src1.replace(OLD_PFX, NEW_PFX)

cells[1]['source'] = src1.splitlines(keepends=True)

# ── Cell[7]: Preload ──────────────────────────────────────────────────────────
src7 = ''.join(cells[7]['source'])

OLD_PRELOAD = "# v08d28: preload from v08d19 (same split)\n"
NEW_PRELOAD = "# v08d32: preload from v08d19 (same split)\n"
assert OLD_PRELOAD in src7
src7 = src7.replace(OLD_PRELOAD, NEW_PRELOAD)

OLD_PRINT = "    print(f'v08d28 preload: {len(DECISION_ROWS_DF)} decisions from {_pfx}')\n"
NEW_PRINT = "    print(f'v08d32 preload: {len(DECISION_ROWS_DF)} decisions from {_pfx}')\n"
assert OLD_PRINT in src7
src7 = src7.replace(OLD_PRINT, NEW_PRINT)

OLD_FAIL = "    print('v08d28 preload: no cache found, mining from scratch')\n"
NEW_FAIL = "    print('v08d32 preload: no cache found, mining from scratch')\n"
assert OLD_FAIL in src7
src7 = src7.replace(OLD_FAIL, NEW_FAIL)

cells[7]['source'] = src7.splitlines(keepends=True)

# ── Cell[19]: Add evolution state + resource features ─────────────────────────
src19 = ''.join(cells[19]['source'])

OLD_CANDS = (
    "        _extra_numeric_candidates = ['supporter_played', 'stadium_played', 'energy_attached',\n"
    "                                      'my_discard_count', 'op_discard_count', 'prize_gap', 'steps_since_op',\n"
    "                                      'position_winrate']\n"
)
NEW_CANDS = (
    "        # v08d32: evolution state + opponent resources + damage estimate\n"
    "        # Good variance confirmed in UNKNOWN_0: top values 8-37% (vs v08d31's 66-83%)\n"
    "        _extra_numeric_candidates = ['supporter_played', 'stadium_played', 'energy_attached',\n"
    "                                      'my_discard_count', 'op_discard_count', 'prize_gap', 'steps_since_op',\n"
    "                                      'position_winrate',\n"
    "                                      'my_alakazam_count', 'my_kadabra_count', 'my_abra_count',\n"
    "                                      'op_bench_count', 'op_hand_count',\n"
    "                                      'powerful_hand_damage_est', 'turn_action_count', 'my_bench_count']\n"
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

assert 'v0_08d32_evo_state' in s1
assert "RUN_PREFIX = 'pokemon-20260627-v0-08d32'" in s1
assert 'v08d19' in s7
assert 'my_alakazam_count' in s19
assert 'my_kadabra_count' in s19
assert 'op_hand_count' in s19
assert 'powerful_hand_damage_est' in s19
assert 'turn_action_count' in s19
assert 'position_winrate' in s19, 'position_winrate must be kept from v08d28'
assert 'op_last_context' in s19
assert '_winner_weight = 4.0' in s19
full = ''.join(''.join(c['source']) for c in nb2['cells'])
assert 'lambdarank' in full
print('All sanity checks passed.')
