"""Create v05d21 from v05d19:
Hypothesis: option_index is a positional shortcut that prevents the model from
learning deeper card-ID / game-state patterns. Remove it and see if winner_margin
improves when the model is forced to discriminate based on actual card IDs.

Changes:
- Exclude ['turn_action_count', 'option_index'] from features
- LambdaRank (same as d19, reverting d20's phase-weight changes)
- Keep clean params from d19 (500 rounds, early_stopping 30)
"""
import json, uuid
from pathlib import Path

WORK_DIR = Path('/kaggle/working')
SRC_NB = WORK_DIR / 'pokemon-20260625-v0-05d19.ipynb'
DST_NB = WORK_DIR / 'pokemon-20260625-v0-05d21.ipynb'

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
src1 = src1.replace("EXPERIMENT_NAME = 'v0_05d19_lgbm_lambdarank'",
                    "EXPERIMENT_NAME = 'v0_05d21_lgbm_lrank_no_optidx'")
src1 = src1.replace("RUN_PREFIX = 'pokemon-20260625-v0-05d19'",
                    "RUN_PREFIX = 'pokemon-20260625-v0-05d21'")
set_cell_src(1, src1)
assert 'v0_05d21_lgbm_lrank_no_optidx' in cell_src(1)
assert 'pokemon-20260625-v0-05d21' in cell_src(1)
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

# ── Patch: exclude option_index in addition to turn_action_count ──────────────
OLD_EXCLUDE = "# Exclude features that can't be reliably computed in inference\nUNKNOWN0_LGBM_EXCLUDE_FEATURES = ['turn_action_count']"
NEW_EXCLUDE = (
    "# Exclude option_index (positional shortcut that crowds out card-ID learning)\n"
    "# and turn_action_count (always 0 in inference = train/test mismatch)\n"
    "UNKNOWN0_LGBM_EXCLUDE_FEATURES = ['turn_action_count', 'option_index']"
)
assert OLD_EXCLUDE in src19, f"OLD_EXCLUDE not found!"
src19 = src19.replace(OLD_EXCLUDE, NEW_EXCLUDE)

# ── Patch: update model name in report ────────────────────────────────────────
OLD_MODEL_NAME = "                    'model': 'lightgbm_lambdarank',"
NEW_MODEL_NAME = "                    'model': 'lightgbm_lambdarank_no_optidx',"
assert OLD_MODEL_NAME in src19, "OLD_MODEL_NAME not found!"
src19 = src19.replace(OLD_MODEL_NAME, NEW_MODEL_NAME)

set_cell_src(cell19_idx, src19)

c19 = cell_src(cell19_idx)
assert "'option_index'" in c19, "option_index not referenced!"
assert "EXCLUDE_FEATURES = ['turn_action_count', 'option_index']" in c19
assert "lightgbm_lambdarank_no_optidx" in c19
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
print(f"\nv05d21 notebook written: {DST_NB}")
print(f"Total cells: {len(nb['cells'])}")
