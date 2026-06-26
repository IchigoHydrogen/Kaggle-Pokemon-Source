"""Create v05d30 from v05d25:
Signature-aware routing: the 4 negative-margin signatures hurt more than they help.

Per-signature analysis from d25 holdout:
  END|N1|NO         lgbm=-0.231  rule=-0.071  → rule is better
  END|N3p|YES       lgbm=-0.115  rule=+0.020  → rule is MUCH better
  END|N3p|NUMBER|YES lgbm=-0.135  rule=-0.102 → rule is better
  END|N1|NO|YES     lgbm=-0.057  rule=+0.077  → rule is MUCH better

Three-layer fix:
1. Training: downweight bad-sig decisions (0.2x) so model focuses on good sigs
2. Offline eval: override LGBM pred with rule_proxy_score for bad sigs → true combined metric
3. Inference _LGBM_INJ_CODE: return None for bad sigs → policy table / rule fallback
"""
import json, uuid
from pathlib import Path

WORK_DIR = Path('/kaggle/working')
SRC_NB = WORK_DIR / 'pokemon-20260625-v0-05d25.ipynb'
DST_NB = WORK_DIR / 'pokemon-20260625-v0-05d30.ipynb'

RULE_FALLBACK_SIGS = {'END|N1|NO', 'END|N3p|YES', 'END|N3p|NUMBER|YES', 'END|N1|NO|YES'}

with open(SRC_NB) as f:
    nb = json.load(f)

def cell_src(i):
    return ''.join(nb['cells'][i]['source'])

def set_cell_src(i, s):
    nb['cells'][i]['source'] = s
    nb['cells'][i]['outputs'] = []
    if nb['cells'][i].get('cell_type') == 'code':
        nb['cells'][i]['execution_count'] = None

# ── Cell 1: version bump ─────────────────────────────────────────────────────
src1 = cell_src(1)
src1 = src1.replace("EXPERIMENT_NAME = 'v0_05d25_lgbm_winner_wt4'",
                    "EXPERIMENT_NAME = 'v0_05d30_lgbm_sig_routing'")
src1 = src1.replace("RUN_PREFIX = 'pokemon-20260625-v0-05d25'",
                    "RUN_PREFIX = 'pokemon-20260625-v0-05d30'")
set_cell_src(1, src1)
assert 'v0_05d30_lgbm_sig_routing' in cell_src(1)
assert 'pokemon-20260625-v0-05d30' in cell_src(1)
print("Cell 1: OK")

# ── Find Cell 19 ─────────────────────────────────────────────────────────────
cell19_idx = None
for i, cell in enumerate(nb['cells']):
    src = ''.join(cell['source'])
    if 'lambdarank' in src and 'UNKNOWN0_MLP_REPORT' in src:
        cell19_idx = i
        break
if cell19_idx is None:
    raise RuntimeError("Could not find Cell 19")
print(f"Cell 19 found at index {cell19_idx}")

src19 = cell_src(cell19_idx)

# ── Patch 1: Downweight bad-sig decisions in training (0.2x) ─────────────────
OLD_WINNER_WT = (
    "            # Winner-weighted training: winner decisions get 2x weight\n"
    "            if 'won' in train_df.columns:\n"
    "                _winner_weight = 4.0\n"
    "                train_df = train_df.copy()\n"
    "                train_df['_win_wt'] = train_df['won'].fillna(0).astype(float) * (_winner_weight - 1.0) + 1.0\n"
    "                _n_winner = int((train_df['won'] == True).sum())\n"
    "                print(f'Winner-weighted training: {_n_winner}/{len(train_df)} winner rows (weight={_winner_weight}x)')\n"
    "            else:\n"
    "                train_df = train_df.copy()\n"
    "                train_df['_win_wt'] = 1.0\n"
    "                print('won column not found, using uniform weights')"
)
NEW_WINNER_WT = (
    "            # Winner-weighted training with signature quality downweighting\n"
    "            _RULE_FALLBACK_SIGS = {'END|N1|NO', 'END|N3p|YES', 'END|N3p|NUMBER|YES', 'END|N1|NO|YES'}\n"
    "            train_df = train_df.copy()\n"
    "            if 'won' in train_df.columns:\n"
    "                _winner_weight = 4.0\n"
    "                train_df['_win_wt'] = train_df['won'].fillna(0).astype(float) * (_winner_weight - 1.0) + 1.0\n"
    "                # Downweight bad-sig decisions: model hurts there, reduce their gradient\n"
    "                if 'option_signature' in train_df.columns:\n"
    "                    _bad_mask = train_df['option_signature'].isin(_RULE_FALLBACK_SIGS)\n"
    "                    train_df.loc[_bad_mask, '_win_wt'] *= 0.2\n"
    "                    _n_bad = int(_bad_mask.sum())\n"
    "                    _n_winner = int((train_df['won'] == True).sum())\n"
    "                    print(f'Sig routing: {_n_bad} bad-sig rows downweighted (0.2x), {_n_winner} winner rows')\n"
    "            else:\n"
    "                train_df['_win_wt'] = 1.0\n"
    "                print('won column not found, using uniform weights')"
)
assert OLD_WINNER_WT in src19, "OLD_WINNER_WT not found!"
src19 = src19.replace(OLD_WINNER_WT, NEW_WINNER_WT)

# ── Patch 2: Override LGBM pred with rule for bad sigs in offline eval ────────
OLD_PRED_LINE = "                pred_df['unknown0_mlp_pred'] = _lgbm_pred  # keep col name for downstream compat"
NEW_PRED_LINE = (
    "                pred_df['unknown0_mlp_pred'] = _lgbm_pred  # keep col name for downstream compat\n"
    "                # Sig routing: override LGBM with rule_proxy_score for bad sigs\n"
    "                _RULE_FALLBACK_SIGS_EVAL = {'END|N1|NO', 'END|N3p|YES', 'END|N3p|NUMBER|YES', 'END|N1|NO|YES'}\n"
    "                if 'option_signature' in pred_df.columns and 'rule_proxy_score' in pred_df.columns:\n"
    "                    _rule_mask = pred_df['option_signature'].isin(_RULE_FALLBACK_SIGS_EVAL)\n"
    "                    pred_df.loc[_rule_mask, 'unknown0_mlp_pred'] = pred_df.loc[_rule_mask, 'rule_proxy_score']\n"
    "                    _n_rule_decs = pred_df[_rule_mask]['decision_id'].nunique()\n"
    "                    print(f'Sig routing eval: rule overrides {_n_rule_decs} decisions from bad sigs')"
)
assert OLD_PRED_LINE in src19, "OLD_PRED_LINE not found!"
src19 = src19.replace(OLD_PRED_LINE, NEW_PRED_LINE)

# ── Patch 3: _LGBM_INJ_CODE — add sig routing (return None for bad sigs) ─────
# The inference code repr stores \n as \\n and ' as \'
# We insert after the `sig = ...` line in the repr'd _LGBM_INJ_CODE string
OLD_SIG_LINE = (
    "sig = _unknown0_policy_abstract_sig(select) if \\\'USE_ABSTRACT_OPTION_SIGNATURE\\\' in globals()"
    " and USE_ABSTRACT_OPTION_SIGNATURE else \\\'N0\\\'\\n"
    "        rows = []"
)
NEW_SIG_LINE = (
    "sig = _unknown0_policy_abstract_sig(select) if \\\'USE_ABSTRACT_OPTION_SIGNATURE\\\' in globals()"
    " and USE_ABSTRACT_OPTION_SIGNATURE else \\\'N0\\\'\\n"
    "        if sig in frozenset({\\\'END|N1|NO\\\', \\\'END|N3p|YES\\\', \\\'END|N3p|NUMBER|YES\\\', \\\'END|N1|NO|YES\\\'}):\\n"
    "            return None  # rule fallback: model hurts on these sigs\\n"
    "        rows = []"
)
assert OLD_SIG_LINE in src19, f"OLD_SIG_LINE not found in Cell 19 source!"
src19 = src19.replace(OLD_SIG_LINE, NEW_SIG_LINE)

# ── Patch 4: model name ───────────────────────────────────────────────────────
OLD_MODEL_NAME = "                    'model': 'lightgbm_lambdarank_winner_wt4',"
NEW_MODEL_NAME = "                    'model': 'lightgbm_lambdarank_sig_routing',"
assert OLD_MODEL_NAME in src19, "OLD_MODEL_NAME not found!"
src19 = src19.replace(OLD_MODEL_NAME, NEW_MODEL_NAME)

set_cell_src(cell19_idx, src19)

c19 = cell_src(cell19_idx)
assert '_RULE_FALLBACK_SIGS' in c19, "_RULE_FALLBACK_SIGS not in Cell 19!"
assert 'sig_routing' in c19, "model name not updated!"
assert 'return None  # rule fallback' in c19 or "return None" in c19, "inference routing not added!"
assert '_rule_mask' in c19, "eval routing not added!"
print("Cell 19: OK")

for cell in nb['cells']:
    if 'id' not in cell:
        cell['id'] = str(uuid.uuid4())[:8]
for cell in nb['cells']:
    if cell.get('cell_type') == 'code':
        cell['outputs'] = []
        cell['execution_count'] = None

with open(DST_NB, 'w') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print(f"\nv05d30 notebook written: {DST_NB}")
