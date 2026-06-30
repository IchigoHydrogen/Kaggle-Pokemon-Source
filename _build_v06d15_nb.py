"""Build pokemon-20260624-v0-06d15.ipynb from v0-06d12 with winner-conditioned disagreement diagnostics."""
import json
import os

SRC = "/home/ei/wkdir/2422005/kaggle/working/pokemon-20260624-v0-06d12/pokemon-20260624-v0-06d12.ipynb"
DST = "/home/ei/wkdir/2422005/kaggle/working/pokemon-20260624-v0-06d15/pokemon-20260624-v0-06d15.ipynb"
CELL_A = "/home/ei/wkdir/2422005/kaggle/working/_v06d15_main_cell.py"
CELL_B = "/home/ei/wkdir/2422005/kaggle/working/_v06d15_runtime_cell.py"

nb = json.load(open(SRC, encoding="utf-8"))
cell_a_src = open(CELL_A, encoding="utf-8").read()
cell_b_src = open(CELL_B, encoding="utf-8").read()

for cell in nb["cells"]:
    if cell.get("cell_type") == "code":
        cell["outputs"] = []
        cell["execution_count"] = None

nb["cells"][0]["source"] = [
    "# v0-06d15 Winner-conditioned rule-disagreement diagnostics\n",
    "\n",
    "Learning diagnostic version. Keeps v0-06d12/v06d14 per-decision CE loss, "
    "card metadata features, model architecture, deck, deterministic split, and no "
    "new runtime action changes. Adds winner-conditioned rule/log disagreement reports "
    "to prioritize the next action family after END deferral.\n",
]

config = "".join(nb["cells"][1]["source"])
config = config.replace(
    "EXPERIMENT_NAME = 'v0_06d12_ce_card_meta_extended_guard'",
    "EXPERIMENT_NAME = 'v0_06d15_winner_rule_disagreement_diagnostic'",
)
config = config.replace(
    "RUN_PREFIX = os.environ.get('V05_RUN_PREFIX', 'pokemon-20260624-v0-06d12')",
    "RUN_PREFIX = os.environ.get('V05_RUN_PREFIX', 'pokemon-20260624-v0-06d15')",
)
nb["cells"][1]["source"] = [line + "\n" for line in config.splitlines()]

config_joined = "".join(nb["cells"][1]["source"])
assert "pokemon-20260624-v0-06d15" in config_joined
assert "v0_06d15_winner_rule_disagreement_diagnostic" in config_joined

section_header_idx = None
part_a_idx = None
part_b_md_idx = None
part_b_code_idx = None
for i, cell in enumerate(nb["cells"]):
    src = "".join(cell.get("source", []))
    ct = cell.get("cell_type")
    if ct == "markdown" and "v0-06d12:" in src:
        section_header_idx = i
    if ct == "code" and "v0-06d12 Part A" in src:
        part_a_idx = i
    if ct == "markdown" and "v0-06d12 Part B" in src:
        part_b_md_idx = i
    if ct == "code" and "v0-06d12 Part B" in src:
        part_b_code_idx = i

if section_header_idx is None or part_a_idx is None or part_b_code_idx is None:
    raise RuntimeError(
        f"could not locate v06d12 cells: section={section_header_idx} "
        f"part_a={part_a_idx} part_b={part_b_code_idx}"
    )

nb["cells"][section_header_idx]["source"] = [
    "## v0-06d15: Winner-conditioned rule-disagreement diagnostics\n",
    "\n",
    "**Part A** — Training and diagnostics: Keeps v0-06d12 CE/card metadata/model "
    "formulation, then adds winner-conditioned rule/log disagreement reports, transition "
    "matrices, family summaries, and rank slices. This is prioritization evidence, not "
    "causal proof.\n",
    "\n",
    "**Part B** — Runtime: Builds a shadow_torch_policy archive, validates Kaggle loader "
    "entrypoint, and runs local smoke eval with latency and shadow-disagreement counters. "
    "Model output does not change actions in v06d15.\n",
]

cells_to_remove = {part_a_idx, part_b_code_idx}
if part_b_md_idx is not None:
    cells_to_remove.add(part_b_md_idx)
nb["cells"] = [c for i, c in enumerate(nb["cells"]) if i not in cells_to_remove]

insert_before = None
for i, cell in enumerate(nb["cells"]):
    src = "".join(cell.get("source", []))
    if cell.get("cell_type") == "markdown" and "v0-06d15: Winner-conditioned" in src:
        insert_before = i + 1
        break
if insert_before is None:
    raise RuntimeError("could not locate v06d15 insertion point")

part_a_code = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [line + "\n" for line in cell_a_src.rstrip().splitlines()],
}
part_b_markdown = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## v0-06d15 Part B: Shadow Runtime Probe + Kaggle Loader Check\n",
        "\n",
        "Injects trained MAIN scorer into rule-only main.py for shadow measurement only. "
        "Builds `{RUN_PREFIX}-submission.tar.gz` with `models/main_option_scorer.pt`, "
        "validates `_kaggle_submission_entrypoint`, and runs smoke eval with latency plus "
        "shadow disagreement counts.\n",
    ],
}
part_b_code = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [line + "\n" for line in cell_b_src.rstrip().splitlines()],
}
nb["cells"].insert(insert_before, part_a_code)
nb["cells"].insert(insert_before + 1, part_b_markdown)
nb["cells"].insert(insert_before + 2, part_b_code)

final_idx = len(nb["cells"]) - 1
final_src = "".join(nb["cells"][final_idx]["source"])
final_src = final_src.replace("v06d12_main_hybrid_report", "v06d15_main_hybrid_report")
final_src = final_src.replace("v06d12_main_hybrid_status", "v06d15_main_hybrid_status")
final_src = final_src.replace("v06d12_runtime_probe", "v06d15_runtime_probe")
final_src = final_src.replace("v06d12_promotion", "v06d15_promotion")
nb["cells"][final_idx]["source"] = [line + "\n" for line in final_src.splitlines()]

errors = []
for i, cell in enumerate(nb["cells"]):
    if cell.get("cell_type") == "code":
        src = "".join(cell.get("source", []))
        try:
            compile(src, f"cell_{i}", "exec")
        except SyntaxError as e:
            errors.append(f"cell_{i}: {e}")
if errors:
    raise RuntimeError("Syntax errors in generated notebook:\n" + "\n".join(errors))

os.makedirs(os.path.dirname(DST), exist_ok=True)
json.dump(nb, open(DST, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"wrote {DST}")
print(f"cells total: {len(nb['cells'])}")
