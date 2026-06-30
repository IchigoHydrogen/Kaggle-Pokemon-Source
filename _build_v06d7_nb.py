"""Build pokemon-20260623-v0-06d7.ipynb from v0-06d6 + MAIN learning cell."""
import json


SRC = "/kaggle/working/pokemon-20260623-v0-06d6.ipynb"
DST = "/kaggle/working/pokemon-20260623-v0-06d7.ipynb"
CELL = "/kaggle/working/_v06d7_main_learning_cell.py"

nb = json.load(open(SRC))
main_src = open(CELL).read()

nb["cells"][0]["source"] = [
    "# v0-06d7 MAIN Replay-Supervised Learning Notebook\n",
    "\n",
    "This notebook continues from the v0-06d6 replay-agreement diagnostic line. It keeps the v0-05d6-derived rule/UNKNOWN_0 gameplay baseline, then adds an offline PyTorch option scorer trained on real replay `MAIN` decisions for the Alakazam actor. The model is evaluated offline only; submission runtime adoption is intentionally disabled in this version.\n",
]

config = "".join(nb["cells"][1]["source"])
config = config.replace(
    "EXPERIMENT_NAME = 'v0_06d6_replay_agreement_diagnostic'",
    "EXPERIMENT_NAME = 'v0_06d7_main_replay_supervised_learning'",
)
config = config.replace(
    "RUN_PREFIX = os.environ.get('V05_RUN_PREFIX', 'pokemon-20260623-v0-06d6')",
    "RUN_PREFIX = os.environ.get('V05_RUN_PREFIX', 'pokemon-20260623-v0-06d7')",
)
nb["cells"][1]["source"] = [line + "\n" for line in config.splitlines()]

for cell in nb["cells"]:
    if cell.get("cell_type") != "code":
        continue
    src = "".join(cell.get("source", []))
    if "v0-06d6 replay-agreement diagnostic" not in src:
        continue
    src = src.replace(
        "    mod_path = OUTPUT_DIR / 'pokemon-20260623-v0-06d6-replay_agreement_agent.py'\n"
        "    if 'write_text' in globals():\n"
        "        write_text(mod_path, src)\n"
        "    else:\n"
        "        open(mod_path, 'w').write(src)\n",
        "    mod_path = OUTPUT_DIR / (RUN_PREFIX + '-replay_agreement_agent.py')\n"
        "    mod_path.write_text(src, encoding='utf-8')\n",
    )
    src = src.replace(
        "spec = _ilu.spec_from_file_location('v06d6_agreement_agent', str(mod_path))",
        "spec = _ilu.spec_from_file_location('v06d7_replay_agreement_agent', str(mod_path))",
    )
    cell["source"] = [line + "\n" for line in src.splitlines()]

final_idx = len(nb["cells"]) - 1
final_src = "".join(nb["cells"][final_idx]["source"])
attach = """
if 'MAIN_LEARNING_REPORT' in globals():
    V05_RUN_SUMMARY['v06d7_main_learning_report'] = MAIN_LEARNING_REPORT
    V05_RUN_SUMMARY['v06d7_main_learning_status'] = MAIN_LEARNING_REPORT.get('status')
    write_json(OUTPUT_DIR / 'v05_run_summary.json', V05_RUN_SUMMARY)
"""
if "v06d7_main_learning_report" not in final_src:
    marker = "write_json(OUTPUT_DIR / 'v05_errors.json', V05_ERROR_ROWS)\n"
    final_src = final_src.replace(marker, marker + attach, 1)
nb["cells"][final_idx]["source"] = [line + "\n" for line in final_src.splitlines()]

md = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## v0-06d7: MAIN Replay-Supervised Offline PyTorch Scorer\n",
        "\n",
        "Builds a real replay supervised `MAIN` dataset, trains a compact PyTorch option scorer, and compares it to the v0-05d6/v0-06d6 rule baseline on valid/holdout. This version does not enable torch in the submission archive.\n",
    ],
}
code = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [line + "\n" for line in main_src.rstrip().splitlines()],
}

nb["cells"] = nb["cells"][:final_idx] + [md, code] + nb["cells"][final_idx:]

for i, cell in enumerate(nb["cells"]):
    if cell.get("cell_type") == "code":
        compile("".join(cell.get("source", [])), f"cell_{i}", "exec")

json.dump(nb, open(DST, "w"), ensure_ascii=False, indent=1)
print("wrote", DST)
print("cells", len(nb["cells"]))
