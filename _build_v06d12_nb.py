"""Build pokemon-20260624-v0-06d12.ipynb from v0-06d11 + per-decision CE + card metadata + extended guard scope."""
import json

SRC = "/home/ei/wkdir/2422005/kaggle/working/pokemon-20260624-v0-06d11.ipynb"
DST = "/home/ei/wkdir/2422005/kaggle/working/pokemon-20260624-v0-06d12/pokemon-20260624-v0-06d12.ipynb"
CELL_A = "/home/ei/wkdir/2422005/kaggle/working/_v06d12_main_cell.py"
CELL_B = "/home/ei/wkdir/2422005/kaggle/working/_v06d12_runtime_cell.py"

nb = json.load(open(SRC))
cell_a_src = open(CELL_A).read()
cell_b_src = open(CELL_B).read()

# Clear all cell outputs and execution counts
for cell in nb["cells"]:
    if cell.get("cell_type") == "code":
        cell["outputs"] = []
        cell["execution_count"] = None

# Cell 0: title markdown
nb["cells"][0]["source"] = [
    "# v0-06d12 Per-decision CE + Card metadata features + Extended guard scope\n",
    "\n",
    "Infrastructure redesign fixing three compounding implementation errors: "
    "(1) BCE loss → per-decision cross-entropy (F.cross_entropy over n options per decision), "
    "(2) card_id%32 one-hot hash (32-bucket collision) → card metadata from all_card_data() "
    "(cardType, evolution stage, ex/mega/tera/aceSpec flags, HP, 19 key-card identity bits, 35 dims), "
    "(3) ATTACK/END/RETREAT hard veto → confidence-gated override (ABILITY remains hard-vetoed). "
    "Feature dim stays at 96. Same MLP architecture, same episode split, fresh training from scratch.\n",
]

# Cell 1: config — update RUN_PREFIX and EXPERIMENT_NAME
config = "".join(nb["cells"][1]["source"])
config = config.replace(
    "EXPERIMENT_NAME = 'v0_06d11_loader_entrypoint_fix'",
    "EXPERIMENT_NAME = 'v0_06d12_ce_card_meta_extended_guard'",
)
config = config.replace(
    "RUN_PREFIX = os.environ.get('V05_RUN_PREFIX', 'pokemon-20260624-v0-06d11')",
    "RUN_PREFIX = os.environ.get('V05_RUN_PREFIX', 'pokemon-20260624-v0-06d12')",
)
nb["cells"][1]["source"] = [line + "\n" for line in config.splitlines()]

config_joined = "".join(nb["cells"][1]["source"])
assert "v0-06d12" in config_joined, "RUN_PREFIX update failed"
assert "v0_06d12_ce_card_meta_extended_guard" in config_joined, "EXPERIMENT_NAME update failed"
print("cell 1 config updated")

# Find the section header markdown (cell 22 in v06d11) and Part A/B cells
# Section header: markdown cell with "v0-06d11" + training section header (not Part B)
section_header_idx = None
part_a_idx = None
part_b_md_idx = None
part_b_code_idx = None

for i, cell in enumerate(nb["cells"]):
    src = "".join(cell.get("source", []))
    ct = cell.get("cell_type")
    if ct == "markdown" and "v0-06d11" in src and "Part B" not in src and "Part A" not in src:
        # This is the section header markdown (cell 22)
        section_header_idx = i
        print(f"found section header at cell {i}: {src[:80].strip()}")
    if ct == "code" and "v0-06d11 Part A" in src:
        part_a_idx = i
        print(f"found Part A code at cell {i}")
    if ct == "markdown" and "v0-06d11 Part B" in src:
        part_b_md_idx = i
        print(f"found Part B markdown at cell {i}")
    if ct == "code" and "v0-06d11 Part B" in src:
        part_b_code_idx = i
        print(f"found Part B code at cell {i}")

if section_header_idx is None:
    raise RuntimeError("could not find v0-06d11 section header markdown")
if part_a_idx is None:
    raise RuntimeError("could not find v0-06d11 Part A code cell")

# Update section header markdown to v06d12
nb["cells"][section_header_idx]["source"] = [
    "## v0-06d12: Per-decision CE + Card metadata + Extended guard scope\n",
    "\n",
    "**Part A** — Training: Fixes three compounding errors in one version. "
    "(1) Loss: BCEWithLogitsLoss → F.cross_entropy over options within each decision. "
    "(2) Features: card_id%32 one-hot (32-bucket collision) → 35-dim card metadata "
    "from all_card_data() at [61-95], feature_dim=96 unchanged. "
    "(3) Hybrid guard: ATTACK/END/RETREAT hard veto removed, now confidence-gated "
    "(ABILITY remains hard-vetoed). Threshold grid on valid split.\n",
    "\n",
    "**Part B** — Runtime: Builds guarded_torch_policy archive, validates Kaggle loader "
    "entrypoint (_kaggle_submission_entrypoint must be last callable), "
    "runs local smoke eval with latency measurement.\n",
]
print(f"updated section header at cell {section_header_idx}")

# Remove Part A code + Part B markdown + Part B code
cells_to_remove = {part_a_idx}
if part_b_md_idx is not None:
    cells_to_remove.add(part_b_md_idx)
if part_b_code_idx is not None:
    cells_to_remove.add(part_b_code_idx)
print(f"removing cells: {sorted(cells_to_remove)}")

nb["cells"] = [c for i, c in enumerate(nb["cells"]) if i not in cells_to_remove]

# After removal: find the updated section header to locate insertion point
insert_before = None
for i, cell in enumerate(nb["cells"]):
    if cell.get("cell_type") != "markdown":
        continue
    src = "".join(cell.get("source", []))
    if "v0-06d12: Per-decision CE" in src:
        insert_before = i + 1
        print(f"found updated section header at cell {i}, inserting at {insert_before}")
        break

if insert_before is None:
    raise RuntimeError("could not find updated section header after removal")

# Build new Part A, Part B markdown, Part B code cells
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
        "## v0-06d12 Part B: Guarded Runtime Probe + Kaggle Loader Check\n",
        "\n",
        "Injects trained MAIN scorer (CE + card metadata features) into rule-only main.py. "
        "Guard: PLAY/ATTACH/EVOLVE/RETREAT enabled (ATTACK/END excluded: model < rule; ABILITY hard-vetoed). "
        "Builds `{RUN_PREFIX}-submission.tar.gz` with `models/main_option_scorer.pt`. "
        "Validates _kaggle_submission_entrypoint is last callable. Smoke eval with latency.\n",
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
print(f"inserted Part A at {insert_before}, Part B md at {insert_before+1}, Part B code at {insert_before+2}")

# Update final summary cell
final_idx = len(nb["cells"]) - 1
final_src = "".join(nb["cells"][final_idx]["source"])
for old, new in [
    ("V05_RUN_SUMMARY['v06d11_main_hybrid_report'] = MAIN_HYBRID_REPORT",
     "V05_RUN_SUMMARY['v06d12_main_hybrid_report'] = MAIN_HYBRID_REPORT"),
    ("V05_RUN_SUMMARY['v06d11_main_hybrid_status'] = MAIN_HYBRID_REPORT.get('status')",
     "V05_RUN_SUMMARY['v06d12_main_hybrid_status'] = MAIN_HYBRID_REPORT.get('status')"),
]:
    final_src = final_src.replace(old, new)
if "v06d12_runtime_probe" not in final_src:
    promo_attach = """
if 'MAIN_HYBRID_REPORT' in globals() and isinstance(MAIN_HYBRID_REPORT, dict):
    V05_RUN_SUMMARY['v06d12_main_hybrid_report'] = MAIN_HYBRID_REPORT
    V05_RUN_SUMMARY['v06d12_main_hybrid_status'] = MAIN_HYBRID_REPORT.get('status')
    V05_RUN_SUMMARY['v06d12_runtime_probe'] = MAIN_HYBRID_REPORT.get('runtime_probe', {})
    V05_RUN_SUMMARY['v06d12_promotion'] = MAIN_HYBRID_REPORT.get('all_gates_ok', False)
    write_json(OUTPUT_DIR / 'v05_run_summary.json', V05_RUN_SUMMARY)
"""
    marker = "write_json(OUTPUT_DIR / 'v05_errors.json', V05_ERROR_ROWS)\n"
    if marker in final_src:
        final_src = final_src.replace(marker, marker + promo_attach, 1)
nb["cells"][final_idx]["source"] = [line + "\n" for line in final_src.splitlines()]

# Syntax check all code cells
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
code_count = sum(1 for c in nb["cells"] if c.get("cell_type") == "code")
print(f"all {code_count} code cells pass syntax check")

import os
os.makedirs(os.path.dirname(DST), exist_ok=True)
json.dump(nb, open(DST, "w"), ensure_ascii=False, indent=1)
print(f"wrote {DST}")
print(f"cells total: {len(nb['cells'])}")
