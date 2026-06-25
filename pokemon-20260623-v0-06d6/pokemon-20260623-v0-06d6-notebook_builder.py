"""Build pokemon-20260623-v0-06d6.ipynb from v0-05d6 + the replay-agreement cell."""
import json
import re

SRC = "/kaggle/working/pokemon-20260623-v0-05d6.ipynb"
DST = "/kaggle/working/pokemon-20260623-v0-06d6.ipynb"
CELL = "/kaggle/working/_v06d6_agreement_cell.py"

nb = json.load(open(SRC))
agree_src = open(CELL).read()

# locate code cells by index in the cells list
code_idx = [i for i, c in enumerate(nb["cells"]) if c["cell_type"] == "code"]

# --- cell 0: bump experiment name + run prefix ---
c0 = nb["cells"][code_idx[0]]
s0 = "".join(c0["source"])
s0 = s0.replace("v0_05d6_unknown0_holdout_free_policy_fit", "v0_06d6_replay_agreement_diagnostic")
s0 = s0.replace("'pokemon-20260623-v0-05d6'", "'pokemon-20260623-v0-06d6'")
assert "pokemon-20260623-v0-06d6" in s0, "prefix bump failed"
c0["source"] = s0

# --- patch run-summary cell (last code cell, has V05_RUN_SUMMARY) to include agreement ---
last_code = code_idx[-1]
cs = "".join(nb["cells"][last_code]["source"])
assert "V05_RUN_SUMMARY" in cs, "run summary cell not found at last code cell"
# insert agreement summary into the run summary dict via a post-construction patch line.
patch = (
    "\n# v0-06d6: attach replay-agreement diagnostic to the run summary\n"
    "try:\n"
    "    V05_RUN_SUMMARY['replay_agreement_summary'] = REPLAY_AGREEMENT_SUMMARY if 'REPLAY_AGREEMENT_SUMMARY' in globals() else {'status': 'missing'}\n"
    "except Exception as _e:\n"
    "    pass\n"
)
# Append patch at end of the run-summary cell source, before any final write if present.
# Simplest robust approach: append to the end of the cell.
nb["cells"][last_code]["source"] = cs + patch

# --- insert the agreement cell right before the run-summary cell ---
agree_cell = {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": agree_src}
nb["cells"].insert(last_code, agree_cell)

json.dump(nb, open(DST, "w"), ensure_ascii=False, indent=1)
print("wrote", DST)
print("code cells now:", sum(1 for c in nb["cells"] if c["cell_type"] == "code"))
