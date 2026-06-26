"""Create v05d23 from v05d22:
1. Add rule_proxy_score as a training feature (meta-stacking).
   The rule proxy is a domain-expert heuristic. Adding it lets LightGBM learn
   'when to follow the rule vs override it'. In d22, rule_proxy_top1≈0.31 on
   END|N3p but LGBM=0.44. The model can now learn 'prefer LGBM over rule when
   the rule disagrees with the chosen action'.
2. Increase num_leaves 127 → 255 (more model capacity).
3. Keep winner-only training from d22.
"""
import json, uuid
from pathlib import Path

WORK_DIR = Path('/kaggle/working')
SRC_NB = WORK_DIR / 'pokemon-20260625-v0-05d22.ipynb'
DST_NB = WORK_DIR / 'pokemon-20260625-v0-05d23.ipynb'

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
src1 = src1.replace("EXPERIMENT_NAME = 'v0_05d22_lgbm_winner_only'",
                    "EXPERIMENT_NAME = 'v0_05d23_lgbm_rule_meta'")
src1 = src1.replace("RUN_PREFIX = 'pokemon-20260625-v0-05d22'",
                    "RUN_PREFIX = 'pokemon-20260625-v0-05d23'")
set_cell_src(1, src1)
assert 'v0_05d23_lgbm_rule_meta' in cell_src(1)
assert 'pokemon-20260625-v0-05d23' in cell_src(1)
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

# ── Patch 1: compute rule_proxy_score for training frames ─────────────────────
OLD_SPLIT_END = (
    "            if train_df.empty or valid_df.empty or train_df['is_chosen'].nunique() < 2:\n"
    "                UNKNOWN0_MLP_REPORT = {'status': 'skipped', 'reason': 'bad_unknown0_split'}"
)
NEW_SPLIT_END = (
    "            # Add rule_proxy_score as meta-feature for stacking\n"
    "            try:\n"
    "                train_df = train_df.copy()\n"
    "                valid_df = valid_df.copy()\n"
    "                train_df['rule_proxy_score'] = offline_rule_proxy_scores(train_df)\n"
    "                valid_df['rule_proxy_score'] = offline_rule_proxy_scores(valid_df)\n"
    "                _has_rule_proxy = True\n"
    "                print(f'rule_proxy_score added to train ({train_df[\"rule_proxy_score\"].notna().sum()} non-null)')\n"
    "            except Exception as _rpe:\n"
    "                _has_rule_proxy = False\n"
    "                print(f'rule_proxy_score unavailable: {_rpe}')\n"
    "\n"
    "            if train_df.empty or valid_df.empty or train_df['is_chosen'].nunique() < 2:\n"
    "                UNKNOWN0_MLP_REPORT = {'status': 'skipped', 'reason': 'bad_unknown0_split'}"
)
assert OLD_SPLIT_END in src19, "OLD_SPLIT_END not found!"
src19 = src19.replace(OLD_SPLIT_END, NEW_SPLIT_END)

# ── Patch 2: add rule_proxy_score to all_feats ────────────────────────────────
OLD_ALL_FEATS = (
    "                _all_feats = _all_feats + [f for f in _extra_numeric_avail if f not in _all_feats]"
)
NEW_ALL_FEATS = (
    "                _all_feats = _all_feats + [f for f in _extra_numeric_avail if f not in _all_feats]\n"
    "                # Add rule_proxy_score as meta-feature if available\n"
    "                if _has_rule_proxy and 'rule_proxy_score' not in _all_feats:\n"
    "                    _all_feats = _all_feats + ['rule_proxy_score']"
)
assert OLD_ALL_FEATS in src19, "OLD_ALL_FEATS not found!"
src19 = src19.replace(OLD_ALL_FEATS, NEW_ALL_FEATS)

# ── Patch 3: increase num_leaves 127 → 255 ────────────────────────────────────
OLD_LEAVES = "                    'num_leaves': 127,"
NEW_LEAVES = "                    'num_leaves': 255,"
assert OLD_LEAVES in src19, "OLD_LEAVES not found!"
src19 = src19.replace(OLD_LEAVES, NEW_LEAVES)

# ── Patch 4: update model name ─────────────────────────────────────────────────
OLD_MODEL_NAME = "                    'model': 'lightgbm_lambdarank_winner_only',"
NEW_MODEL_NAME = "                    'model': 'lightgbm_lambdarank_rule_meta',"
assert OLD_MODEL_NAME in src19, "OLD_MODEL_NAME not found!"
src19 = src19.replace(OLD_MODEL_NAME, NEW_MODEL_NAME)

# ── Patch 5: add num_leaves to report ──────────────────────────────────────────
OLD_LEAVES_REPORT = "                    'num_leaves': int(_lgbm_params['num_leaves']),"
NEW_LEAVES_REPORT = (
    "                    'num_leaves': int(_lgbm_params['num_leaves']),\n"
    "                    'rule_proxy_as_feature': _has_rule_proxy,"
)
assert OLD_LEAVES_REPORT in src19, "OLD_LEAVES_REPORT not found!"
src19 = src19.replace(OLD_LEAVES_REPORT, NEW_LEAVES_REPORT)

# ── Patch 6: save extra_feats + rule_proxy to prep dict ──────────────────────
# Rule proxy also needs to be noted in prep so inference code knows to skip it
# (rule_proxy_score is not available in inference main.py; if it's in feature_names,
# the encoding loop will default to 0.0 which is the "rule disagrees strongly" signal)
# This is acceptable: at inference, rule_proxy=0 means "follow LGBM instinct"

set_cell_src(cell19_idx, src19)

c19 = cell_src(cell19_idx)
assert 'rule_proxy_score' in c19, "rule_proxy_score not in Cell 19!"
assert "num_leaves': 255" in c19, "num_leaves not updated!"
assert 'lightgbm_lambdarank_rule_meta' in c19, "model name not updated!"
assert 'offline_rule_proxy_scores(train_df)' in c19, "train rule proxy not added!"
print("Cell 19: OK")

# ── Add IDs / clear outputs ───────────────────────────────────────────────────
for cell in nb['cells']:
    if 'id' not in cell:
        cell['id'] = str(uuid.uuid4())[:8]
for cell in nb['cells']:
    if cell.get('cell_type') == 'code':
        cell['outputs'] = []
        cell['execution_count'] = None

with open(DST_NB, 'w') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print(f"\nv05d23 notebook written: {DST_NB}")
print(f"Total cells: {len(nb['cells'])}")
