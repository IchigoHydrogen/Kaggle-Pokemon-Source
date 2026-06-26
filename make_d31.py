"""Create v05d31 from v05d25:
Signature routing — fix d30's two bugs:
  1. eval override was inserted BEFORE rule_proxy_score was assigned (line 748 vs 749)
     → silently skipped; now placed correctly AFTER rule_proxy_score exists
  2. training downweighting (0.2x) was too aggressive; caused END|N1 to regress
     → removed; only pure routing at eval + inference

Expected gain: bad sigs (19.8% of holdout) improve from lgbm=-0.123 → rule=-0.015,
giving +0.021 to overall winner_margin (0.0555 → ~0.077).
"""
import json, uuid
from pathlib import Path

WORK_DIR = Path('/kaggle/working')
SRC_NB = WORK_DIR / 'pokemon-20260625-v0-05d25.ipynb'
DST_NB = WORK_DIR / 'pokemon-20260625-v0-05d31.ipynb'

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
                    "EXPERIMENT_NAME = 'v0_05d31_lgbm_sig_route_fix'")
src1 = src1.replace("RUN_PREFIX = 'pokemon-20260625-v0-05d25'",
                    "RUN_PREFIX = 'pokemon-20260625-v0-05d31'")
set_cell_src(1, src1)
assert 'v0_05d31_lgbm_sig_route_fix' in cell_src(1)
assert 'pokemon-20260625-v0-05d31' in cell_src(1)
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

# ── Patch 1: Eval override — insert AFTER rule_proxy_score is available ───────
# Line 748: pred_df['unknown0_mlp_pred'] = _lgbm_pred
# Line 749: pred_df['rule_proxy_score'] = offline_rule_proxy_scores(pred_df)
# We must insert AFTER line 749
OLD_PRED_BLOCK = (
    "                pred_df['unknown0_mlp_pred'] = _lgbm_pred  # keep col name for downstream compat\n"
    "                pred_df['rule_proxy_score'] = offline_rule_proxy_scores(pred_df)"
)
NEW_PRED_BLOCK = (
    "                pred_df['unknown0_mlp_pred'] = _lgbm_pred  # keep col name for downstream compat\n"
    "                pred_df['rule_proxy_score'] = offline_rule_proxy_scores(pred_df)\n"
    "                # Sig routing eval: override LGBM with rule for negative-margin signatures\n"
    "                _RULE_FALLBACK_SIGS_EVAL = {'END|N1|NO', 'END|N3p|YES', 'END|N3p|NUMBER|YES', 'END|N1|NO|YES'}\n"
    "                if 'option_signature' in pred_df.columns:\n"
    "                    _rule_mask = pred_df['option_signature'].isin(_RULE_FALLBACK_SIGS_EVAL)\n"
    "                    pred_df.loc[_rule_mask, 'unknown0_mlp_pred'] = pred_df.loc[_rule_mask, 'rule_proxy_score']\n"
    "                    _n_rule_decs = pred_df[_rule_mask]['decision_id'].nunique()\n"
    "                    print(f'Sig routing eval: rule overrides {_n_rule_decs} decisions (sigs: {sorted(_RULE_FALLBACK_SIGS_EVAL)})')"
)
assert OLD_PRED_BLOCK in src19, "OLD_PRED_BLOCK not found!"
src19 = src19.replace(OLD_PRED_BLOCK, NEW_PRED_BLOCK)

# ── Patch 2: _LGBM_INJ_CODE — sig routing at inference ───────────────────────
# After `sig = _unknown0_policy_abstract_sig(...)`, add return None for bad sigs
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
assert OLD_SIG_LINE in src19, "OLD_SIG_LINE not found!"
src19 = src19.replace(OLD_SIG_LINE, NEW_SIG_LINE)

# ── Patch 3: model name ───────────────────────────────────────────────────────
OLD_MODEL_NAME = "                    'model': 'lightgbm_lambdarank_winner_wt4',"
NEW_MODEL_NAME = "                    'model': 'lightgbm_lambdarank_sig_route_fix',"
assert OLD_MODEL_NAME in src19, "OLD_MODEL_NAME not found!"
src19 = src19.replace(OLD_MODEL_NAME, NEW_MODEL_NAME)

set_cell_src(cell19_idx, src19)

c19 = cell_src(cell19_idx)
assert '_RULE_FALLBACK_SIGS_EVAL' in c19, "_RULE_FALLBACK_SIGS_EVAL not in Cell 19!"
assert 'sig_route_fix' in c19, "model name not updated!"
assert 'return None  # rule fallback' in c19, "inference routing not added!"
assert "rule overrides" in c19, "eval routing message not added!"
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
print(f"\nv05d31 notebook written: {DST_NB}")
