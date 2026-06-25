"""Build pokemon-20260624-v0-06d10.ipynb from v0-06d9 + early stopping + guarded runtime cell."""
import json

SRC = "/kaggle/working/pokemon-20260624-v0-06d9.ipynb"
DST = "/kaggle/working/pokemon-20260624-v0-06d10.ipynb"
CELL_A = "/kaggle/working/_v06d10_main_hybrid_cell.py"
CELL_B = "/kaggle/working/_v06d10_guarded_runtime_cell.py"

nb = json.load(open(SRC))
cell_a_src = open(CELL_A).read()
cell_b_src = open(CELL_B).read()

# Cell 0: title markdown
nb["cells"][0]["source"] = [
    "# v0-06d10 Early Stopping + Guarded Runtime Probe\n",
    "\n",
    "Part A: Early stopping on valid top-1 (patience=5, max_epochs=100) with best checkpoint restore. "
    "Part B: Injects trained MAIN scorer into rule-only main.py as a guarded_torch_policy "
    "(PLAY/ATTACH/EVOLVE safe overrides; ABILITY/RETREAT/ATTACK/END hard veto). "
    "Builds submission archive with models/main_option_scorer.pt and runs local smoke eval "
    "with latency tracking.\n",
]

# Cell 1: config — update RUN_PREFIX and EXPERIMENT_NAME
config = "".join(nb["cells"][1]["source"])
config = config.replace(
    "EXPERIMENT_NAME = 'v0_06d9_vectorized_gpu_preload'",
    "EXPERIMENT_NAME = 'v0_06d10_early_stop_guarded_runtime'",
)
config = config.replace(
    "RUN_PREFIX = os.environ.get('V05_RUN_PREFIX', 'pokemon-20260624-v0-06d9')",
    "RUN_PREFIX = os.environ.get('V05_RUN_PREFIX', 'pokemon-20260624-v0-06d10')",
)
nb["cells"][1]["source"] = [line + "\n" for line in config.splitlines()]

# Clear all cell outputs and execution counts (prevent stale outputs from v0-06d9)
for cell in nb["cells"]:
    if cell.get("cell_type") == "code":
        cell["outputs"] = []
        cell["execution_count"] = None

# Cell 22: update markdown cell just before the MAIN training cell
for i, cell in enumerate(nb["cells"]):
    if cell.get("cell_type") != "markdown":
        continue
    src = "".join(cell.get("source", []))
    if "v0-06d9: Vectorized Feature Extraction" in src:
        nb["cells"][i]["source"] = [
            "## v0-06d10: Early Stopping + Guarded Runtime Probe\n",
            "\n",
            "**Part A** — Adds early stopping on valid top-1 accuracy (patience=5, max_epochs=100) "
            "with best-epoch checkpoint restore to the MAIN replay-supervised training loop. "
            "Same features, architecture, and threshold selection as v0-06d9.\n",
            "\n",
            "**Part B** — Injects the trained MAIN scorer into the rule-only main.py as a "
            "guarded_torch_policy agent: PLAY/ATTACH/EVOLVE overrides enabled; "
            "ABILITY/RETREAT/ATTACK/END hard-vetoed. Builds `{RUN_PREFIX}-submission.tar.gz` "
            "with `models/main_option_scorer.pt` and runs local smoke eval with "
            "per-decision latency measurement.\n",
        ]
        print(f"updated markdown cell {i}")
        break

# Cell 23: replace v0-06d9 MAIN training cell with Part A (early stopping)
replaced_a = False
for i, cell in enumerate(nb["cells"]):
    if cell.get("cell_type") != "code":
        continue
    src = "".join(cell.get("source", []))
    if "v0-06d9 MAIN replay-supervised guarded hybrid" in src:
        nb["cells"][i]["source"] = [line + "\n" for line in cell_a_src.rstrip().splitlines()]
        print(f"replaced cell {i} with v0-06d10 Part A (early stopping training)")
        replaced_a = True
        part_a_cell_idx = i
        break

if not replaced_a:
    raise RuntimeError("could not find v0-06d9 MAIN training cell to replace")

# Insert Part B cells (markdown + code) right after Part A
part_b_markdown = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## v0-06d10 Part B: Guarded Runtime Probe\n",
        "\n",
        "Injects the trained MAIN scorer into the rule-only main.py. "
        "Builds `{RUN_PREFIX}-submission.tar.gz` (guarded_torch_policy) with "
        "`models/main_option_scorer.pt`. "
        "Runs local smoke eval and measures per-decision inference latency.\n",
    ],
}
part_b_code = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [line + "\n" for line in cell_b_src.rstrip().splitlines()],
}

insert_pos = part_a_cell_idx + 1
nb["cells"].insert(insert_pos, part_b_markdown)
nb["cells"].insert(insert_pos + 1, part_b_code)
print(f"inserted Part B markdown at {insert_pos} and Part B code at {insert_pos + 1}")

# Final summary cell: update v06d9 → v06d10 references
final_idx = len(nb["cells"]) - 1
final_src = "".join(nb["cells"][final_idx]["source"])
final_src = final_src.replace(
    "V05_RUN_SUMMARY['v06d9_main_hybrid_report'] = MAIN_HYBRID_REPORT",
    "V05_RUN_SUMMARY['v06d10_main_hybrid_report'] = MAIN_HYBRID_REPORT",
)
final_src = final_src.replace(
    "V05_RUN_SUMMARY['v06d9_main_hybrid_status'] = MAIN_HYBRID_REPORT.get('status')",
    "V05_RUN_SUMMARY['v06d10_main_hybrid_status'] = MAIN_HYBRID_REPORT.get('status')",
)
# Add runtime probe to summary if not already present
if "v06d10_runtime_probe" not in final_src:
    promo_attach = """
if 'MAIN_HYBRID_REPORT' in globals() and isinstance(MAIN_HYBRID_REPORT, dict):
    V05_RUN_SUMMARY['v06d10_main_hybrid_report'] = MAIN_HYBRID_REPORT
    V05_RUN_SUMMARY['v06d10_main_hybrid_status'] = MAIN_HYBRID_REPORT.get('status')
    V05_RUN_SUMMARY['v06d10_runtime_probe'] = MAIN_HYBRID_REPORT.get('runtime_probe', {})
    V05_RUN_SUMMARY['v06d10_promotion'] = MAIN_HYBRID_REPORT.get('all_gates_ok', False)
    write_json(OUTPUT_DIR / 'v05_run_summary.json', V05_RUN_SUMMARY)
"""
    marker = "write_json(OUTPUT_DIR / 'v05_errors.json', V05_ERROR_ROWS)\n"
    if marker in final_src:
        final_src = final_src.replace(marker, marker + promo_attach, 1)
nb["cells"][final_idx]["source"] = [line + "\n" for line in final_src.splitlines()]

# Syntax check all code cells
for i, cell in enumerate(nb["cells"]):
    if cell.get("cell_type") == "code":
        compile("".join(cell.get("source", [])), f"cell_{i}", "exec")

json.dump(nb, open(DST, "w"), ensure_ascii=False, indent=1)
print(f"wrote {DST}")
print(f"cells: {len(nb['cells'])}")
