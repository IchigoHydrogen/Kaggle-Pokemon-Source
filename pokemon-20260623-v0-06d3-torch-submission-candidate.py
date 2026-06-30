from __future__ import annotations

import csv
import importlib.util
import json
import math
import os
import random
import shutil
import statistics
import sys
import tarfile
import tempfile
import time
import traceback
from pathlib import Path


RUN_PREFIX = "pokemon-20260623-v0-06d3"
WORKING_DIR = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path(".")
OUTPUT_DIR = WORKING_DIR / RUN_PREFIX
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BASELINE_ARCHIVE = WORKING_DIR / "pokemon-20260623-v0-06d1-submission.tar.gz"
PROBE_ARCHIVE = WORKING_DIR / f"{RUN_PREFIX}-submission.tar.gz"
MODEL_NAME = f"{RUN_PREFIX}-torch_probe_state.pt"
MODEL_PATH = OUTPUT_DIR / MODEL_NAME
RANDOM_SEED = 1515
TORCH_THREADS = int(os.environ.get("V06_TORCH_THREADS", "1"))
RUN_GAMES = int(os.environ.get("V06_TORCH_PROBE_GAMES", "96"))
MAX_STEPS = int(os.environ.get("V06_TORCH_PROBE_MAX_STEPS", "1000"))
ACT_TIMEOUT_CANDIDATES = [0.25, 0.5, 1.0, 2.0, 3.0]
OVERAGE_BUDGET_SECONDS = 600.0


def write_json(path: Path, obj) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    return str(path)


def write_csv(path: Path, rows: list[dict]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({k for row in rows for k in row.keys()}) if rows else ["empty"]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return str(path)


def write_latency_tables(rows: list[dict]) -> dict:
    csv_path = OUTPUT_DIR / f"{RUN_PREFIX}-latency_rows.csv"
    paths = {"csv": write_csv(csv_path, rows)}
    try:
        import pandas as pd

        parquet_path = OUTPUT_DIR / f"{RUN_PREFIX}-latency_rows.parquet"
        pd.DataFrame(rows).to_parquet(parquet_path, index=False)
        paths["parquet"] = str(parquet_path)
    except Exception as exc:
        paths["parquet_error"] = repr(exc)
    return paths


def percentile(values: list[float], q: float):
    if not values:
        return None
    vals = sorted(values)
    if len(vals) == 1:
        return float(vals[0])
    pos = (len(vals) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(vals[lo])
    return float(vals[lo] * (hi - pos) + vals[hi] * (pos - lo))


def summarize_durations(values: list[float]) -> dict:
    vals = [float(v) for v in values if isinstance(v, (int, float))]
    return {
        "count": int(len(vals)),
        "mean": float(statistics.mean(vals)) if vals else None,
        "p50": percentile(vals, 0.50),
        "p90": percentile(vals, 0.90),
        "p95": percentile(vals, 0.95),
        "p99": percentile(vals, 0.99),
        "max": float(max(vals)) if vals else None,
        "sum": float(sum(vals)) if vals else 0.0,
    }


def find_cg_dir() -> Path:
    candidates = [
        Path("/kaggle/input/competitions/pokemon-tcg-ai-battle/sample_submission/cg"),
        Path("/kaggle/input/pokemon-tcg-ai-battle/sample_submission/cg"),
        Path("input/competitions/pokemon-tcg-ai-battle/sample_submission/cg"),
    ]
    for c in candidates:
        if c.exists() and (c / "api.py").exists():
            return c
    raise FileNotFoundError("Could not find cg directory")


def ensure_cg_importable() -> dict:
    status = {"available": False, "cg_dir": None, "cg_parent": None, "error": ""}
    try:
        import cg.api  # noqa: F401
        import cg.game  # noqa: F401
        status["available"] = True
        return status
    except Exception as exc:
        status["error"] = repr(exc)
    cg_dir = find_cg_dir()
    status["cg_dir"] = str(cg_dir)
    status["cg_parent"] = str(cg_dir.parent)
    if str(cg_dir.parent) not in sys.path:
        sys.path.insert(0, str(cg_dir.parent))
    try:
        import cg.api  # noqa: F401
        import cg.game  # noqa: F401
        status["available"] = True
        status["error"] = ""
    except Exception as exc:
        status["error"] = repr(exc)
    return status


def extract_baseline() -> tuple[str, list[int]]:
    if not BASELINE_ARCHIVE.exists():
        raise FileNotFoundError(f"baseline archive not found: {BASELINE_ARCHIVE}")
    with tarfile.open(BASELINE_ARCHIVE, "r:gz") as tar:
        main_source = tar.extractfile("main.py").read().decode("utf-8")
        deck_text = tar.extractfile("deck.csv").read().decode("utf-8").strip().splitlines()
    deck = [int(row.strip().split(",")[0]) for row in deck_text if row.strip()]
    if len(deck) != 60:
        raise AssertionError(f"baseline deck length must be 60, got {len(deck)}")
    return main_source, deck


TORCH_PROBE_SOURCE = r'''

# v0-06d3 production-form PyTorch shadow-mode submission candidate.
# This model is intentionally not used to change actions yet. It is loaded and
# executed on every non-deck agent call to measure feasibility under real main.py.
TORCH_PROBE_STATS = {
    "available": False,
    "load_error": "",
    "calls": 0,
    "inference_calls": 0,
    "options_scored": 0,
    "inference_seconds": 0.0,
    "max_inference_seconds": 0.0,
    "feature_dim": 64,
    "param_count": 0,
}

try:
    import time as _torch_probe_time
    import torch
    import torch.nn as nn

    try:
        torch.set_num_threads(int(os.environ.get("POKEMON_TORCH_THREADS", "1")))
    except Exception:
        pass

    class _TorchProbeNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(64, 512),
                nn.ReLU(),
                nn.Linear(512, 384),
                nn.ReLU(),
                nn.Linear(384, 256),
                nn.ReLU(),
                nn.Linear(256, 1),
            )

        def forward(self, x):
            return self.net(x).squeeze(-1)

    _TORCH_PROBE_MODEL = _TorchProbeNet()
    _TORCH_PROBE_MODEL.load_state_dict(torch.load(os.path.join(os.path.dirname(__file__), "__MODEL_NAME__"), map_location="cpu"))
    _TORCH_PROBE_MODEL.eval()
    TORCH_PROBE_STATS["param_count"] = int(sum(p.numel() for p in _TORCH_PROBE_MODEL.parameters()))
    TORCH_PROBE_STATS["available"] = True
except Exception as _torch_probe_exc:
    TORCH_PROBE_STATS["load_error"] = repr(_torch_probe_exc)
    _TORCH_PROBE_MODEL = None


def _torch_probe_num(x, default=0.0):
    try:
        if x is None:
            return float(default)
        return float(x)
    except Exception:
        return float(default)


def _torch_probe_enum(x):
    try:
        v = getattr(x, "value", None)
        if v is not None:
            return int(v)
    except Exception:
        pass
    try:
        return int(x)
    except Exception:
        return abs(hash(str(x))) % 97


def _torch_probe_card_id(obs, option):
    try:
        area = getattr(option, "area", None)
        idx = getattr(option, "index", None)
        player = getattr(option, "playerIndex", None)
        if area is None or idx is None or player is None:
            return 0
        card = get_card(obs, area, int(idx), int(player))
        return int(getattr(card, "id", 0) or 0)
    except Exception:
        return 0


def _torch_probe_features(obs):
    if not TORCH_PROBE_STATS.get("available"):
        return None
    try:
        select = obs.select
        opts = list(getattr(select, "option", []) or [])
        n = len(opts)
        if n <= 0:
            return None
        rows = []
        state = getattr(obs, "current", None)
        your_index = _torch_probe_num(getattr(state, "yourIndex", 0), 0.0)
        turn = _torch_probe_num(getattr(state, "turn", 0), 0.0)
        step = _torch_probe_num(getattr(state, "step", 0), 0.0)
        context_id = _torch_probe_enum(getattr(select, "context", 0))
        min_count = _torch_probe_num(getattr(select, "minCount", 0), 0.0)
        max_count = _torch_probe_num(getattr(select, "maxCount", 0), 0.0)
        for i, opt in enumerate(opts):
            row = [0.0] * 64
            opt_type = _torch_probe_enum(getattr(opt, "type", 0))
            card_id = _torch_probe_card_id(obs, opt)
            row[0] = 1.0
            row[1] = min(1.0, n / 32.0)
            row[2] = min_count / 8.0
            row[3] = max_count / 8.0
            row[4] = min(1.0, turn / 16.0)
            row[5] = min(1.0, step / 512.0)
            row[6] = your_index
            row[7] = min(1.0, i / max(1, n - 1))
            row[8] = (context_id % 37) / 37.0
            row[9] = (opt_type % 23) / 23.0
            row[10] = (card_id % 2048) / 2048.0
            row[11] = _torch_probe_num(getattr(opt, "number", 0), 0.0) / 16.0
            row[12] = _torch_probe_num(getattr(opt, "index", 0), 0.0) / 16.0
            row[13] = _torch_probe_num(getattr(opt, "inPlayIndex", 0), 0.0) / 16.0
            row[14 + (opt_type % 16)] = 1.0
            row[30 + (context_id % 16)] = 1.0
            row[46 + (card_id % 16)] = 1.0
            rows.append(row)
        return torch.tensor(rows, dtype=torch.float32)
    except Exception:
        return None


def _torch_probe_infer(obs):
    TORCH_PROBE_STATS["calls"] = TORCH_PROBE_STATS.get("calls", 0) + 1
    x = _torch_probe_features(obs)
    if x is None:
        return None
    t0 = _torch_probe_time.perf_counter()
    try:
        with torch.no_grad():
            y = _TORCH_PROBE_MODEL(x)
        dt = _torch_probe_time.perf_counter() - t0
        TORCH_PROBE_STATS["inference_calls"] += 1
        TORCH_PROBE_STATS["options_scored"] += int(x.shape[0])
        TORCH_PROBE_STATS["inference_seconds"] += float(dt)
        TORCH_PROBE_STATS["max_inference_seconds"] = max(float(TORCH_PROBE_STATS["max_inference_seconds"]), float(dt))
        return y
    except Exception as exc:
        TORCH_PROBE_STATS["load_error"] = "inference:" + repr(exc)
        return None
'''


def build_probe_model():
    import torch
    import torch.nn as nn

    torch.manual_seed(RANDOM_SEED)
    torch.set_num_threads(TORCH_THREADS)

    class TorchProbeNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(64, 512),
                nn.ReLU(),
                nn.Linear(512, 384),
                nn.ReLU(),
                nn.Linear(384, 256),
                nn.ReLU(),
                nn.Linear(256, 1),
            )

        def forward(self, x):
            return self.net(x).squeeze(-1)

    model = TorchProbeNet()
    param_count = int(sum(p.numel() for p in model.parameters()))
    if not (300_000 <= param_count <= 500_000):
        raise AssertionError(f"param_count out of target range: {param_count}")
    torch.save(model.state_dict(), MODEL_PATH)
    return {"param_count": param_count, "model_path": str(MODEL_PATH)}


def inject_torch_probe(main_source: str) -> str:
    probe = TORCH_PROBE_SOURCE.replace("__MODEL_NAME__", MODEL_NAME)
    marker = "def agent(obs_dict: dict) -> list[int]:\n"
    if marker not in main_source:
        raise AssertionError("agent function marker not found")
    source = main_source.replace(marker, probe + "\n\n" + marker, 1)
    needle = "    if obs.select is None:\n        return my_deck\n"
    if needle not in source:
        raise AssertionError("agent deck-return block not found")
    replacement = needle + "    _torch_probe_infer(obs)\n"
    source = source.replace(needle, replacement, 1)
    compile(source, "main.py", "exec")
    if "import torch" not in source:
        raise AssertionError("torch import was not injected")
    return source


def exclude_python_cache(tarinfo: tarfile.TarInfo):
    name = tarinfo.name
    parts = Path(name).parts
    if "__pycache__" in parts or name.endswith((".pyc", ".pyo")):
        return None
    return tarinfo


def build_probe_archive(main_source: str, deck: list[int]) -> dict:
    tmp = OUTPUT_DIR / "archive_build"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True, exist_ok=True)
    (tmp / "main.py").write_text(main_source, encoding="utf-8")
    with open(tmp / "deck.csv", "w", encoding="utf-8", newline="") as f:
        for x in deck:
            f.write(f"{int(x)}\n")
    shutil.copy2(MODEL_PATH, tmp / MODEL_NAME)
    cg_dir = find_cg_dir()
    with tarfile.open(PROBE_ARCHIVE, "w:gz") as tar:
        tar.add(tmp / "main.py", arcname="main.py", filter=exclude_python_cache)
        tar.add(tmp / "deck.csv", arcname="deck.csv", filter=exclude_python_cache)
        tar.add(tmp / MODEL_NAME, arcname=MODEL_NAME, filter=exclude_python_cache)
        tar.add(str(cg_dir), arcname="cg", filter=exclude_python_cache)
    with tarfile.open(PROBE_ARCHIVE, "r:gz") as tar:
        names = tar.getnames()
    required = {"main.py", "deck.csv", "cg/api.py", "cg/game.py", MODEL_NAME}
    return {
        "archive": str(PROBE_ARCHIVE),
        "archive_size_bytes": int(PROBE_ARCHIVE.stat().st_size),
        "files": int(len(names)),
        "required_present": sorted(required.intersection(set(names))),
        "missing_required": sorted(required.difference(set(names))),
        "has_pycache": any("__pycache__" in n for n in names),
        "has_pyc_or_pyo": any(n.endswith((".pyc", ".pyo")) for n in names),
        "has_torch_model": MODEL_NAME in names,
        "preview": names[:25],
    }


def import_agent_from_source(main_source: str, label: str):
    path = OUTPUT_DIR / f"tmp_{label}_{time.time_ns()}_{random.randint(0, 10**9)}.py"
    path.write_text(main_source, encoding="utf-8")
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    old_path = list(sys.path)
    sys.path.insert(0, str(OUTPUT_DIR))
    try:
        t0 = time.perf_counter()
        spec.loader.exec_module(module)
        import_seconds = time.perf_counter() - t0
    finally:
        sys.path[:] = old_path
    return module, import_seconds


def make_random_agent(deck: list[int]):
    def random_agent(obs_dict):
        from cg.api import to_observation_class

        obs = to_observation_class(obs_dict)
        if obs.select is None:
            return deck
        max_count = int(obs.select.maxCount)
        min_count = int(obs.select.minCount)
        idxs = list(range(len(obs.select.option)))
        if max_count <= 0:
            return []
        k = max(min_count, min(max_count, len(idxs)))
        if k >= len(idxs):
            return idxs[:k]
        return random.sample(idxs, k)

    return random_agent


HOP_CONTROL_FALLBACK_DECK = (
    [878] * 4 + [879] * 4 + [311] * 3 + [304] * 2 + [1092] * 1 +
    [1115] * 4 + [1134] * 4 + [1122] * 4 + [1097] * 1 + [1152] * 2 +
    [1197] * 2 + [1171] * 4 + [1219] * 4 + [1227] * 4 + [1182] * 2 +
    [1225] * 3 + [1255] * 4 + [19] * 4 + [11] * 4
)
LUCARIO_EXP14_FALLBACK_DECK = (
    [673] * 2 + [674] * 2 + [675] * 2 + [676] * 3 + [677] * 3 + [678] * 4 +
    [1102] * 4 + [1123] * 2 + [1141] * 4 + [1142] * 4 + [1152] * 4 +
    [1159] * 1 + [1182] * 2 + [1192] * 4 + [1227] * 4 + [1252] * 2 + [6] * 13
)


def timed_call(fn, obs):
    t0 = time.perf_counter()
    action = fn(obs)
    return action, time.perf_counter() - t0


def play_game(agent_module, deck: list[int], opponent_agent, opponent_deck: list[int], target_seat: int, game_id: int, suite: str, opponent_name: str):
    from cg.api import to_observation_class
    from cg.game import battle_select, battle_start

    latencies = []
    try:
        if target_seat == 0:
            obs, start_data = battle_start(deck, opponent_deck)
        else:
            obs, start_data = battle_start(opponent_deck, deck)
        if obs is None:
            return {"suite": suite, "opponent": opponent_name, "game": game_id, "seat": target_seat, "won": False, "result": None, "steps": 0, "error": f"battle_start_failed:{getattr(start_data, 'errorPlayer', None)}:{getattr(start_data, 'errorType', None)}", "illegal_action": False}, latencies
        for step in range(MAX_STEPS + 1):
            obc = to_observation_class(obs)
            if obc.current.result >= 0:
                won = int(obc.current.result) == target_seat
                return {"suite": suite, "opponent": opponent_name, "game": game_id, "seat": target_seat, "won": bool(won), "result": int(obc.current.result), "steps": step, "error": "", "illegal_action": False}, latencies
            active = int(obc.current.yourIndex)
            if active == target_seat:
                action, dt = timed_call(agent_module.agent, obs)
                latencies.append({"suite": suite, "opponent": opponent_name, "game": game_id, "seat": target_seat, "step": step, "duration": float(dt), "n_options": len(obc.select.option) if obc.select is not None else 0})
            else:
                action = opponent_agent(obs)
            if not isinstance(action, list) or any(not isinstance(x, int) for x in action):
                return {"suite": suite, "opponent": opponent_name, "game": game_id, "seat": target_seat, "won": False, "result": None, "steps": step, "error": f"illegal_action_type:{action!r}", "illegal_action": True}, latencies
            if obc.select is not None:
                min_count = int(obc.select.minCount)
                max_count = int(obc.select.maxCount)
                nopt = len(obc.select.option)
                if len(action) < min_count or len(action) > max_count or len(set(action)) != len(action) or any((x < 0 or x >= nopt) for x in action):
                    return {"suite": suite, "opponent": opponent_name, "game": game_id, "seat": target_seat, "won": False, "result": None, "steps": step, "error": f"illegal_action_value:{action!r}:min={min_count}:max={max_count}:nopt={nopt}", "illegal_action": True}, latencies
            obs = battle_select(action)
        return {"suite": suite, "opponent": opponent_name, "game": game_id, "seat": target_seat, "won": False, "result": None, "steps": MAX_STEPS, "error": "max_steps_exceeded", "illegal_action": False}, latencies
    except Exception as exc:
        return {"suite": suite, "opponent": opponent_name, "game": game_id, "seat": target_seat, "won": False, "result": None, "steps": None, "error": repr(exc), "traceback": traceback.format_exc(limit=5), "illegal_action": False}, latencies


def run_durability(main_source: str, deck: list[int]) -> dict:
    status = ensure_cg_importable()
    if not status.get("available"):
        return {"status": "skipped", "reason": "cg_not_available", "cg_import_status": status}

    import_rows = []
    module = None
    for i in range(5):
        module, dt = import_agent_from_source(main_source, f"probe_import_{i}")
        import_rows.append({"iteration": i, "import_seconds": float(dt), "torch_available": bool(getattr(module, "TORCH_PROBE_STATS", {}).get("available")), "param_count": int(getattr(module, "TORCH_PROBE_STATS", {}).get("param_count", 0) or 0), "load_error": str(getattr(module, "TORCH_PROBE_STATS", {}).get("load_error", ""))})

    rows = []
    latency_rows = []
    opponents = [
        ("random_same_deck", make_random_agent(deck), deck),
        ("random_hop_control_deck", make_random_agent(list(HOP_CONTROL_FALLBACK_DECK)), list(HOP_CONTROL_FALLBACK_DECK)),
        ("random_lucario_exp14_deck", make_random_agent(list(LUCARIO_EXP14_FALLBACK_DECK)), list(LUCARIO_EXP14_FALLBACK_DECK)),
    ]
    game_id = 0
    for opponent_name, opponent_agent, opponent_deck in opponents:
        per_opp = max(1, RUN_GAMES // len(opponents))
        for j in range(per_opp):
            seat = game_id % 2
            row, lats = play_game(module, deck, opponent_agent, opponent_deck, seat, game_id, "torch_probe", opponent_name)
            rows.append(row)
            latency_rows.extend(lats)
            game_id += 1

    durations = [r["duration"] for r in latency_rows]
    overage = {}
    for act_timeout in ACT_TIMEOUT_CANDIDATES:
        consumed = sum(max(0.0, float(d) - act_timeout) for d in durations)
        overage[str(act_timeout)] = {
            "total_overage_seconds": float(consumed),
            "budget_seconds": OVERAGE_BUDGET_SECONDS,
            "budget_fraction": float(consumed / OVERAGE_BUDGET_SECONDS),
            "would_exceed_600s_budget": bool(consumed > OVERAGE_BUDGET_SECONDS),
        }

    wins = sum(1 for r in rows if r.get("won"))
    errors = sum(1 for r in rows if r.get("error"))
    illegal = sum(1 for r in rows if r.get("illegal_action"))
    return {
        "status": "ok",
        "cg_import_status": status,
        "import_rows": import_rows,
        "game_summary": {
            "games": int(len(rows)),
            "wins": int(wins),
            "win_rate": float(wins / max(1, len(rows))),
            "errors": int(errors),
            "illegal_action_count": int(illegal),
            "avg_steps": float(statistics.mean([r["steps"] for r in rows if isinstance(r.get("steps"), int)])) if rows else None,
        },
        "latency_summary": summarize_durations(durations),
        "overage_estimates": overage,
        "torch_probe_stats": getattr(module, "TORCH_PROBE_STATS", {}),
        "game_rows_path": write_json(OUTPUT_DIR / f"{RUN_PREFIX}-game_rows.json", rows),
        "latency_rows_paths": write_latency_tables(latency_rows),
        "latency_rows": latency_rows[:5],
    }


def main():
    random.seed(RANDOM_SEED)
    os.environ.setdefault("POKEMON_TORCH_THREADS", str(TORCH_THREADS))
    model_info = build_probe_model()
    baseline_source, deck = extract_baseline()
    probe_source = inject_torch_probe(baseline_source)
    (OUTPUT_DIR / f"{RUN_PREFIX}-main.py").write_text(probe_source, encoding="utf-8")
    archive_summary = build_probe_archive(probe_source, deck)
    durability = run_durability(probe_source, deck)
    report = {
        "version": RUN_PREFIX,
        "baseline": "pokemon-20260623-v0-06d1",
        "purpose": "submit-ready shadow-mode PyTorch runtime archive",
        "model_info": model_info,
        "torch_threads": TORCH_THREADS,
        "run_games_requested": RUN_GAMES,
        "max_steps": MAX_STEPS,
        "archive_summary": archive_summary,
        "durability": durability,
        "decision_inputs": {
            "torch_imported": all(r.get("torch_available") for r in durability.get("import_rows", [])) if durability.get("status") == "ok" else False,
            "param_count_in_range": 300_000 <= int(model_info["param_count"]) <= 500_000,
            "errors": durability.get("game_summary", {}).get("errors"),
            "illegal_action_count": durability.get("game_summary", {}).get("illegal_action_count"),
            "p99_latency": durability.get("latency_summary", {}).get("p99"),
            "max_latency": durability.get("latency_summary", {}).get("max"),
        },
    }
    report_path = OUTPUT_DIR / f"{RUN_PREFIX}-runtime_report.json"
    write_json(report_path, report)
    print(json.dumps(report["decision_inputs"], ensure_ascii=False, indent=2))
    print(str(report_path))


if __name__ == "__main__":
    main()
