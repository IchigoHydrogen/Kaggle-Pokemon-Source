"""Build pokemon-20260624-v0-06d9.ipynb from v0-06d8 + vectorized MAIN hybrid cell."""
import json

SRC = "/kaggle/working/pokemon-20260623-v0-06d8.ipynb"
DST = "/kaggle/working/pokemon-20260624-v0-06d9.ipynb"
CELL = "/kaggle/working/_v06d9_main_hybrid_cell.py"

nb = json.load(open(SRC))
main_src = open(CELL).read()

nb["cells"][0]["source"] = [
    "# v0-06d9 Vectorized Feature Extraction + GPU Pre-loading\n",
    "\n",
    "Infrastructure speedup: NumPy pre-allocated vectorized feature extraction and GPU pre-loading "
    "of training tensors. Same model architecture, feature definition, and hybrid evaluation as v0-06d8. "
    "Cross-check enforced before training.\n",
]

config = "".join(nb["cells"][1]["source"])
config = config.replace(
    "EXPERIMENT_NAME = 'v0_06d8_fresh_main_guarded_hybrid'",
    "EXPERIMENT_NAME = 'v0_06d9_vectorized_gpu_preload'",
)
config = config.replace(
    "RUN_PREFIX = os.environ.get('V05_RUN_PREFIX', 'pokemon-20260623-v0-06d8')",
    "RUN_PREFIX = os.environ.get('V05_RUN_PREFIX', 'pokemon-20260624-v0-06d9')",
)
nb["cells"][1]["source"] = [line + "\n" for line in config.splitlines()]

# Find and replace the v0-06d8 MAIN hybrid code cell (cell 23)
for i, cell in enumerate(nb["cells"]):
    if cell.get("cell_type") != "code":
        continue
    src = "".join(cell.get("source", []))
    if "v0-06d8 MAIN fresh replay-supervised guarded hybrid" in src:
        nb["cells"][i]["source"] = [line + "\n" for line in main_src.rstrip().splitlines()]
        print(f"replaced cell {i} with v0-06d9 hybrid cell")
        break

# Update markdown label before the cell
for i, cell in enumerate(nb["cells"]):
    if cell.get("cell_type") != "markdown":
        continue
    src = "".join(cell.get("source", []))
    if "v0-06d8: Fresh MAIN Replay-Supervised Guarded Hybrid" in src:
        nb["cells"][i]["source"] = [
            "## v0-06d9: Vectorized Feature Extraction + GPU Pre-loading\n",
            "\n",
            "Vectorizes the MAIN option feature extraction (pre-allocated NumPy array + per-decision "
            "slice assignment) and pre-loads all training tensors to GPU before the training loop. "
            "Cross-check verifies vectorized features match the reference Python-loop implementation. "
            "Same model architecture and hybrid evaluation logic as v0-06d8.\n",
        ]
        print(f"updated markdown cell {i}")
        break

# Update final summary cell
final_idx = len(nb["cells"]) - 1
final_src = "".join(nb["cells"][final_idx]["source"])
final_src = final_src.replace(
    "V05_RUN_SUMMARY['v06d8_main_hybrid_report'] = MAIN_HYBRID_REPORT",
    "V05_RUN_SUMMARY['v06d9_main_hybrid_report'] = MAIN_HYBRID_REPORT",
)
final_src = final_src.replace(
    "V05_RUN_SUMMARY['v06d8_main_hybrid_status'] = MAIN_HYBRID_REPORT.get('status')",
    "V05_RUN_SUMMARY['v06d9_main_hybrid_status'] = MAIN_HYBRID_REPORT.get('status')",
)
if "v06d9_main_hybrid_report" not in final_src and "v06d8_main_hybrid_report" not in final_src:
    attach = """
if 'MAIN_HYBRID_REPORT' in globals():
    V05_RUN_SUMMARY['v06d9_main_hybrid_report'] = MAIN_HYBRID_REPORT
    V05_RUN_SUMMARY['v06d9_main_hybrid_status'] = MAIN_HYBRID_REPORT.get('status')
    write_json(OUTPUT_DIR / 'v05_run_summary.json', V05_RUN_SUMMARY)
"""
    marker = "write_json(OUTPUT_DIR / 'v05_errors.json', V05_ERROR_ROWS)\n"
    final_src = final_src.replace(marker, marker + attach, 1)
nb["cells"][final_idx]["source"] = [line + "\n" for line in final_src.splitlines()]

for i, cell in enumerate(nb["cells"]):
    if cell.get("cell_type") == "code":
        compile("".join(cell.get("source", [])), f"cell_{i}", "exec")

json.dump(nb, open(DST, "w"), ensure_ascii=False, indent=1)
print("wrote", DST)
print("cells", len(nb["cells"]))
