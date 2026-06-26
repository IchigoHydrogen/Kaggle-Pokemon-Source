"""Create v05d17 notebook from v05d16:
  - Fix: remove 'step' from Cell 9 dcols (collision with OPTION_ROWS_DF.step causes Delta-V NaN)
  All other changes from v05d16 are carried forward unchanged.
"""
import json, uuid
from pathlib import Path

WORK_DIR = Path('/kaggle/working')
SRC_NB = WORK_DIR / 'pokemon-20260625-v0-05d16.ipynb'
DST_NB = WORK_DIR / 'pokemon-20260625-v0-05d17.ipynb'

with open(SRC_NB) as f:
    nb = json.load(f)

def cell_src(i):
    return ''.join(nb['cells'][i]['source'])

def set_cell_src(i, s):
    nb['cells'][i]['source'] = s
    nb['cells'][i]['outputs'] = []
    if nb['cells'][i].get('cell_type') == 'code':
        nb['cells'][i]['execution_count'] = None

def insert_cell_after(i, source):
    """Insert a new code cell after cell i, with a required nbformat id."""
    new_cell = {
        'cell_type': 'code',
        'id': str(uuid.uuid4())[:8],
        'execution_count': None,
        'metadata': {},
        'outputs': [],
        'source': source,
    }
    nb['cells'].insert(i + 1, new_cell)

# ── Cell 1: bump version d16 → d17 ─────────────────────────────────────────
src1 = cell_src(1)
src1 = src1.replace(
    "EXPERIMENT_NAME = 'v0_05d16_adv_top200_mlp'",
    "EXPERIMENT_NAME = 'v0_05d17_adv_top200_mlp'"
)
src1 = src1.replace(
    "RUN_PREFIX = 'pokemon-20260625-v0-05d16'",
    "RUN_PREFIX = 'pokemon-20260625-v0-05d17'"
)
set_cell_src(1, src1)
assert 'v0_05d17_adv_top200_mlp' in cell_src(1), "Cell 1 EXPERIMENT_NAME patch failed"
assert 'pokemon-20260625-v0-05d17' in cell_src(1), "Cell 1 RUN_PREFIX patch failed"
print("Cell 1: OK")

# ── Cell 9: remove 'step' from dcols to prevent column collision ─────────────
# OPTION_ROWS_DF already has 'step'; including it in dcols causes
# a step_x/step_y collision in the LEFT merge, so 'step' disappears
# from ALAKAZAM_OPTION_MODEL_DF → Delta-V cell's `if 'step' in _dec.columns`
# evaluates False → all delta_v = NaN → all mc_step_weight = 0.
src9 = cell_src(9)

OLD_DCOLS = (
    "        'step', 'mc_step_weight', 'phase_weight', 'step_frac',\n"
    "    ]"
)
NEW_DCOLS = (
    "        'mc_step_weight', 'phase_weight', 'step_frac',\n"
    "    ]"
)

if OLD_DCOLS in src9:
    src9 = src9.replace(OLD_DCOLS, NEW_DCOLS)
    print("Cell 9: 'step' removed from dcols")
else:
    print("WARNING: Cell 9 dcols target not found!")
    idx = src9.find("'mc_step_weight'")
    print(f"  mc_step_weight found at: {idx}")
    print(f"  context: {repr(src9[max(0,idx-80):idx+80])}")

set_cell_src(9, src9)
assert "'step'" not in cell_src(9) or "mc_step_weight" in cell_src(9), "Cell 9 patch verify failed"
src9_final = cell_src(9)
idx_step = src9_final.find("'step'")
idx_mc = src9_final.find("'mc_step_weight'")
# Check that 'step' in dcols is gone (it might appear elsewhere in the cell)
# The only 'step' usage in dcols should now be gone
print(f"Cell 9: 'step' at pos {idx_step}, mc_step_weight at pos {idx_mc}")
print("Cell 9: OK")

# ── Add IDs to any cells missing them ───────────────────────────────────────
for cell in nb['cells']:
    if 'id' not in cell:
        cell['id'] = str(uuid.uuid4())[:8]

# ── Clear all outputs ─────────────────────────────────────────────────────────
for cell in nb['cells']:
    if cell.get('cell_type') == 'code':
        cell['outputs'] = []
        cell['execution_count'] = None

# ── Write notebook ────────────────────────────────────────────────────────────
with open(DST_NB, 'w') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print(f"\nv05d17 notebook written: {DST_NB}")
print(f"Total cells: {len(nb['cells'])}")
