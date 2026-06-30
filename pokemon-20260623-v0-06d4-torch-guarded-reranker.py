from __future__ import annotations

import importlib.util
import json
import math
import os
import random
import statistics
import sys
import tarfile
import time
from pathlib import Path


RUN_PREFIX = "pokemon-20260623-v0-06d4"
WORKING_DIR = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path(".")
OUTPUT_DIR = WORKING_DIR / RUN_PREFIX
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BASELINE_SCRIPT = WORKING_DIR / "pokemon-20260623-v0-06d3-torch-submission-candidate.py"
BASELINE_GAMEPLAY = "pokemon-20260623-v0-06d1"
MODEL_NAME = f"{RUN_PREFIX}-torch_policy_state.pt"
MODEL_PATH = OUTPUT_DIR / MODEL_NAME
RANDOM_SEED = 1617
TORCH_THREADS = int(os.environ.get("V06_TORCH_THREADS", "1"))
TRAIN_GAMES = int(os.environ.get("V06D4_TRAIN_GAMES", "96"))
TRAIN_EPOCHS = int(os.environ.get("V06D4_TRAIN_EPOCHS", "8"))
RUN_GAMES = int(os.environ.get("V06_TORCH_PROBE_GAMES", "96"))


def load_base_module():
    spec = importlib.util.spec_from_file_location("v06d3_base", BASELINE_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.RUN_PREFIX = RUN_PREFIX
    module.OUTPUT_DIR = OUTPUT_DIR
    module.PROBE_ARCHIVE = WORKING_DIR / f"{RUN_PREFIX}-submission.tar.gz"
    module.MODEL_NAME = MODEL_NAME
    module.MODEL_PATH = MODEL_PATH
    module.RUN_GAMES = RUN_GAMES
    return module


base = load_base_module()


def write_json(path: Path, obj) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    return str(path)


def enum_id(x) -> int:
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


def num(x, default=0.0) -> float:
    try:
        if x is None:
            return float(default)
        return float(x)
    except Exception:
        return float(default)


def get_runtime_card_id(obs, option) -> int:
    try:
        card = base.import_agent_from_source  # keep pyflakes quiet when inspected outside Docker
        del card
        area = getattr(option, "area", None)
        idx = getattr(option, "index", None)
        player = getattr(option, "playerIndex", None)
        if area is None or idx is None or player is None:
            return 0
        from cg.api import AreaType

        ps = obs.current.players[int(player)]
        match area:
            case AreaType.DECK:
                c = obs.select.deck[int(idx)]
            case AreaType.HAND:
                c = ps.hand[int(idx)]
            case AreaType.DISCARD:
                c = ps.discard[int(idx)]
            case AreaType.ACTIVE:
                c = ps.active[int(idx)]
            case AreaType.BENCH:
                c = ps.bench[int(idx)]
            case AreaType.PRIZE:
                c = ps.prize[int(idx)]
            case AreaType.STADIUM:
                c = obs.current.stadium[int(idx)]
            case AreaType.LOOKING:
                c = obs.current.looking[int(idx)]
            case _:
                c = None
        return int(getattr(c, "id", 0) or 0)
    except Exception:
        return 0


def option_features(obs, option_index: int, option_count: int) -> list[float]:
    select = obs.select
    option = select.option[option_index]
    state = obs.current
    context_id = enum_id(getattr(select, "context", 0))
    opt_type = enum_id(getattr(option, "type", 0))
    card_id = get_runtime_card_id(obs, option)
    row = [0.0] * 64
    row[0] = 1.0
    row[1] = min(1.0, option_count / 32.0)
    row[2] = num(getattr(select, "minCount", 0)) / 8.0
    row[3] = num(getattr(select, "maxCount", 0)) / 8.0
    row[4] = min(1.0, num(getattr(state, "turn", 0)) / 16.0)
    row[5] = min(1.0, num(getattr(state, "step", 0)) / 512.0)
    row[6] = num(getattr(state, "yourIndex", 0))
    row[7] = min(1.0, option_index / max(1, option_count - 1))
    row[8] = (context_id % 37) / 37.0
    row[9] = (opt_type % 23) / 23.0
    row[10] = (card_id % 2048) / 2048.0
    row[11] = num(getattr(option, "number", 0)) / 16.0
    row[12] = num(getattr(option, "index", 0)) / 16.0
    row[13] = num(getattr(option, "inPlayIndex", 0)) / 16.0
    row[14 + (opt_type % 16)] = 1.0
    row[30 + (context_id % 16)] = 1.0
    row[46 + (card_id % 16)] = 1.0
    return row


def validate_action(action, obs_class) -> str:
    if not isinstance(action, list) or any(not isinstance(x, int) for x in action):
        return f"illegal_action_type:{action!r}"
    if obs_class.select is not None:
        min_count = int(obs_class.select.minCount)
        max_count = int(obs_class.select.maxCount)
        nopt = len(obs_class.select.option)
        if len(action) < min_count or len(action) > max_count:
            return f"illegal_action_count:{action!r}:min={min_count}:max={max_count}"
        if len(set(action)) != len(action) or any((x < 0 or x >= nopt) for x in action):
            return f"illegal_action_value:{action!r}:nopt={nopt}"
    return ""


def collect_training_rows(main_source: str, deck: list[int]) -> dict:
    from cg.api import to_observation_class
    from cg.game import battle_select, battle_start

    teacher, import_seconds = base.import_agent_from_source(main_source, "v06d4_teacher")
    opponents = [
        ("random_same_deck", base.make_random_agent(deck), deck),
        ("random_hop_control_deck", base.make_random_agent(list(base.HOP_CONTROL_FALLBACK_DECK)), list(base.HOP_CONTROL_FALLBACK_DECK)),
        ("random_lucario_exp14_deck", base.make_random_agent(list(base.LUCARIO_EXP14_FALLBACK_DECK)), list(base.LUCARIO_EXP14_FALLBACK_DECK)),
    ]
    rows = []
    decisions = []
    errors = []
    game_id = 0
    for opponent_name, opponent_agent, opponent_deck in opponents:
        per_opp = max(1, TRAIN_GAMES // len(opponents))
        for _ in range(per_opp):
            target_seat = game_id % 2
            if target_seat == 0:
                obs, start_data = battle_start(deck, opponent_deck)
            else:
                obs, start_data = battle_start(opponent_deck, deck)
            if obs is None:
                errors.append({"game_id": game_id, "error": f"battle_start_failed:{getattr(start_data, 'errorType', None)}"})
                game_id += 1
                continue
            for step in range(1000):
                obc = to_observation_class(obs)
                if obc.current.result >= 0:
                    break
                active = int(obc.current.yourIndex)
                if active == target_seat:
                    action = teacher.agent(obs)
                    err = validate_action(action, obc)
                    if err:
                        errors.append({"game_id": game_id, "step": step, "error": err})
                        break
                    if obc.select is not None and len(obc.select.option) > 0:
                        decision_id = len(decisions)
                        selected = set(action)
                        option_count = len(obc.select.option)
                        context_name = str(getattr(getattr(obc.select, "context", None), "name", getattr(obc.select, "context", "")))
                        max_count = int(obc.select.maxCount)
                        min_count = int(obc.select.minCount)
                        for option_index in range(option_count):
                            rows.append({
                                "decision_id": decision_id,
                                "game_id": game_id,
                                "split": "valid" if game_id % 5 == 0 else "train",
                                "option_index": option_index,
                                "label": 1.0 if option_index in selected else 0.0,
                                "features": option_features(obc, option_index, option_count),
                            })
                        decisions.append({
                            "decision_id": decision_id,
                            "game_id": game_id,
                            "split": "valid" if game_id % 5 == 0 else "train",
                            "selected": sorted(selected),
                            "option_count": option_count,
                            "context": context_name,
                            "min_count": min_count,
                            "max_count": max_count,
                            "single_choice": bool(max_count == 1 and min_count <= 1),
                        })
                else:
                    action = opponent_agent(obs)
                    err = validate_action(action, obc)
                    if err:
                        errors.append({"game_id": game_id, "step": step, "error": "opponent:" + err})
                        break
                obs = battle_select(action)
            game_id += 1
    return {
        "teacher_import_seconds": import_seconds,
        "rows": rows,
        "decisions": decisions,
        "errors": errors,
        "summary": {
            "games": game_id,
            "rows": len(rows),
            "decisions": len(decisions),
            "errors": len(errors),
            "train_rows": sum(1 for r in rows if r["split"] == "train"),
            "valid_rows": sum(1 for r in rows if r["split"] == "valid"),
            "single_choice_decisions": sum(1 for d in decisions if d["single_choice"]),
        },
    }


def build_policy_model():
    import torch
    import torch.nn as nn

    torch.manual_seed(RANDOM_SEED)
    torch.set_num_threads(TORCH_THREADS)

    class TorchPolicyNet(nn.Module):
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

    model = TorchPolicyNet()
    param_count = int(sum(p.numel() for p in model.parameters()))
    if not (300_000 <= param_count <= 500_000):
        raise AssertionError(f"param_count out of target range: {param_count}")
    return model, param_count


def train_model(data: dict) -> dict:
    import torch
    import torch.nn as nn

    model, param_count = build_policy_model()
    train_rows = [r for r in data["rows"] if r["split"] == "train"]
    valid_rows = [r for r in data["rows"] if r["split"] == "valid"]
    if len(train_rows) < 100 or len(valid_rows) < 20:
        raise AssertionError(f"not enough training data: train={len(train_rows)} valid={len(valid_rows)}")

    def tensorize(rows):
        x = torch.tensor([r["features"] for r in rows], dtype=torch.float32)
        y = torch.tensor([r["label"] for r in rows], dtype=torch.float32)
        return x, y

    x_train, y_train = tensorize(train_rows)
    x_valid, y_valid = tensorize(valid_rows)
    pos = max(1.0, float(y_train.sum().item()))
    neg = max(1.0, float(len(y_train) - y_train.sum().item()))
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([min(20.0, neg / pos)], dtype=torch.float32))
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    history = []
    for epoch in range(TRAIN_EPOCHS):
        model.train()
        perm = torch.randperm(len(x_train))
        losses = []
        for start in range(0, len(x_train), 1024):
            idx = perm[start:start + 1024]
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(x_train[idx]), y_train[idx])
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
        model.eval()
        with torch.no_grad():
            valid_loss = float(loss_fn(model(x_valid), y_valid).item())
        history.append({"epoch": epoch, "train_loss": float(statistics.mean(losses)), "valid_loss": valid_loss})

    def score_rows(rows):
        model.eval()
        with torch.no_grad():
            x, _ = tensorize(rows)
            return model(x).detach().cpu().tolist()

    def top1_metrics(rows, decisions):
        scores = score_rows(rows)
        by_decision = {}
        for row, score in zip(rows, scores):
            by_decision.setdefault(row["decision_id"], []).append((row["option_index"], float(score), row["label"]))
        hits = 0
        n = 0
        single_hits = 0
        single_n = 0
        decision_map = {d["decision_id"]: d for d in decisions}
        for decision_id, items in by_decision.items():
            pred = max(items, key=lambda x: x[1])[0]
            labels = {idx for idx, _, label in items if label > 0.5}
            hit = pred in labels
            hits += int(hit)
            n += 1
            if decision_map.get(decision_id, {}).get("single_choice"):
                single_hits += int(hit)
                single_n += 1
        return {
            "decisions": n,
            "top1_hit_rate": float(hits / max(1, n)),
            "single_choice_decisions": single_n,
            "single_choice_top1_hit_rate": float(single_hits / max(1, single_n)),
        }

    metrics = {
        "param_count": param_count,
        "train_rows": len(train_rows),
        "valid_rows": len(valid_rows),
        "positive_rate_train": float(y_train.mean().item()),
        "positive_rate_valid": float(y_valid.mean().item()),
        "history": history,
        "train_top1": top1_metrics(train_rows, data["decisions"]),
        "valid_top1": top1_metrics(valid_rows, data["decisions"]),
    }
    torch.save(model.state_dict(), MODEL_PATH)
    metrics["model_path"] = str(MODEL_PATH)
    return metrics


TORCH_POLICY_SOURCE = r'''

# v0-06d4 guarded PyTorch policy reranker.
TORCH_POLICY_STATS = {
    "available": False,
    "load_error": "",
    "calls": 0,
    "eligible": 0,
    "same_top": 0,
    "overrides": 0,
    "skipped": 0,
    "inference_calls": 0,
    "options_scored": 0,
    "inference_seconds": 0.0,
    "max_inference_seconds": 0.0,
    "feature_dim": 64,
    "param_count": 0,
}
TORCH_PROBE_STATS = TORCH_POLICY_STATS

try:
    import time as _torch_policy_time
    import torch
    import torch.nn as nn

    try:
        torch.set_num_threads(int(os.environ.get("POKEMON_TORCH_THREADS", "1")))
    except Exception:
        pass

    class _TorchPolicyNet(nn.Module):
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

    _TORCH_POLICY_MODEL = _TorchPolicyNet()
    _TORCH_POLICY_MODEL.load_state_dict(torch.load(os.path.join(os.path.dirname(__file__), "__MODEL_NAME__"), map_location="cpu"))
    _TORCH_POLICY_MODEL.eval()
    TORCH_POLICY_STATS["param_count"] = int(sum(p.numel() for p in _TORCH_POLICY_MODEL.parameters()))
    TORCH_POLICY_STATS["available"] = True
except Exception as _torch_policy_exc:
    TORCH_POLICY_STATS["load_error"] = repr(_torch_policy_exc)
    _TORCH_POLICY_MODEL = None


def _torch_policy_num(x, default=0.0):
    try:
        if x is None:
            return float(default)
        return float(x)
    except Exception:
        return float(default)


def _torch_policy_enum(x):
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


def _torch_policy_card_id(obs, option):
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


def _torch_policy_features(obs):
    if not TORCH_POLICY_STATS.get("available"):
        return None
    try:
        select = obs.select
        opts = list(getattr(select, "option", []) or [])
        n = len(opts)
        if n <= 0:
            return None
        rows = []
        state = getattr(obs, "current", None)
        your_index = _torch_policy_num(getattr(state, "yourIndex", 0), 0.0)
        turn = _torch_policy_num(getattr(state, "turn", 0), 0.0)
        step = _torch_policy_num(getattr(state, "step", 0), 0.0)
        context_id = _torch_policy_enum(getattr(select, "context", 0))
        min_count = _torch_policy_num(getattr(select, "minCount", 0), 0.0)
        max_count = _torch_policy_num(getattr(select, "maxCount", 0), 0.0)
        for i, opt in enumerate(opts):
            row = [0.0] * 64
            opt_type = _torch_policy_enum(getattr(opt, "type", 0))
            card_id = _torch_policy_card_id(obs, opt)
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
            row[11] = _torch_policy_num(getattr(opt, "number", 0), 0.0) / 16.0
            row[12] = _torch_policy_num(getattr(opt, "index", 0), 0.0) / 16.0
            row[13] = _torch_policy_num(getattr(opt, "inPlayIndex", 0), 0.0) / 16.0
            row[14 + (opt_type % 16)] = 1.0
            row[30 + (context_id % 16)] = 1.0
            row[46 + (card_id % 16)] = 1.0
            rows.append(row)
        return torch.tensor(rows, dtype=torch.float32)
    except Exception:
        return None


def _torch_policy_rerank(obs, selected, scores, context, safe_draws, can_win_this_turn, deckout_risk_strict):
    TORCH_POLICY_STATS["calls"] = TORCH_POLICY_STATS.get("calls", 0) + 1
    try:
        if not TORCH_POLICY_STATS.get("available") or _TORCH_POLICY_MODEL is None:
            TORCH_POLICY_STATS["skipped"] += 1
            return selected
        select = obs.select
        if select is None or not selected:
            TORCH_POLICY_STATS["skipped"] += 1
            return selected
        min_count = max(0, int(select.minCount))
        max_count = min(len(select.option), max(0, int(select.maxCount)))
        if max_count != 1 or min_count > 1 or len(select.option) <= 1:
            TORCH_POLICY_STATS["skipped"] += 1
            return selected
        context_name = str(getattr(context, "name", context))
        context_id = _torch_policy_enum(context)
        allowed_contexts = {"TO_HAND", "TO_ACTIVE", "SWITCH", "TO_BENCH", "ATTACH_FROM"}
        allowed_context_ids = {3, 4, 5, 7, 21}
        if context_name not in allowed_contexts and context_id not in allowed_context_ids:
            TORCH_POLICY_STATS["skipped"] += 1
            return selected
        if deckout_risk_strict and not can_win_this_turn and (context_name == "TO_HAND" or context_id == 7):
            TORCH_POLICY_STATS["skipped"] += 1
            return selected
        base_i = int(selected[0])
        if base_i < 0 or base_i >= len(select.option):
            TORCH_POLICY_STATS["skipped"] += 1
            return selected
        x = _torch_policy_features(obs)
        if x is None or int(x.shape[0]) != len(select.option):
            TORCH_POLICY_STATS["skipped"] += 1
            return selected
        t0 = _torch_policy_time.perf_counter()
        with torch.no_grad():
            y = _TORCH_POLICY_MODEL(x)
        dt = _torch_policy_time.perf_counter() - t0
        TORCH_POLICY_STATS["inference_calls"] += 1
        TORCH_POLICY_STATS["options_scored"] += int(x.shape[0])
        TORCH_POLICY_STATS["inference_seconds"] += float(dt)
        TORCH_POLICY_STATS["max_inference_seconds"] = max(float(TORCH_POLICY_STATS["max_inference_seconds"]), float(dt))
        TORCH_POLICY_STATS["eligible"] += 1
        top_i = int(torch.argmax(y).item())
        if top_i == base_i:
            TORCH_POLICY_STATS["same_top"] += 1
            return selected
        top_logit = float(y[top_i].item())
        base_logit = float(y[base_i].item())
        top_prob = float(torch.sigmoid(y[top_i]).item())
        if top_prob >= 0.72 and (top_logit - base_logit) >= 0.65:
            TORCH_POLICY_STATS["overrides"] += 1
            return safe_unique_action([top_i], len(select.option), min_count, max_count)
        TORCH_POLICY_STATS["skipped"] += 1
        return selected
    except Exception as exc:
        TORCH_POLICY_STATS["load_error"] = "rerank:" + repr(exc)
        TORCH_POLICY_STATS["skipped"] += 1
        return selected
'''


def inject_policy(main_source: str) -> str:
    policy = TORCH_POLICY_SOURCE.replace("__MODEL_NAME__", MODEL_NAME)
    marker = "def agent(obs_dict: dict) -> list[int]:\n"
    if marker not in main_source:
        raise AssertionError("agent marker not found")
    source = main_source.replace(marker, policy + "\n\n" + marker, 1)
    needle = "    if context == SelectContext.MAIN and selected:\n"
    if needle not in source:
        raise AssertionError("main ability flag block not found")
    replacement = "    selected = _torch_policy_rerank(obs, selected, scores, context, safe_draws, can_win_this_turn, deckout_risk_strict)\n\n" + needle
    source = source.replace(needle, replacement, 1)
    compile(source, "main.py", "exec")
    if "import torch" not in source or "_torch_policy_rerank" not in source:
        raise AssertionError("policy injection failed")
    return source


def write_promotion(report: dict) -> dict:
    durability = report.get("durability", {})
    game_summary = durability.get("game_summary", {})
    policy_stats = durability.get("torch_probe_stats", {})
    decision = "promote"
    reasons = []
    if report.get("archive_summary", {}).get("missing_required"):
        decision = "reject"
        reasons.append("archive missing required files")
    if game_summary.get("errors", 1) != 0 or game_summary.get("illegal_action_count", 1) != 0:
        decision = "reject"
        reasons.append("durability errors or illegal actions")
    if not policy_stats.get("available"):
        decision = "reject"
        reasons.append("torch policy unavailable")
    if int(policy_stats.get("overrides", 0) or 0) <= 0:
        decision = "needs_followup"
        reasons.append("no measured action overrides; path is effectively shadow-like")
    if not reasons:
        reasons.append("guarded torch policy changed actions and passed local hard gates")
    obj = {
        "decision": decision,
        "promotion_type": "guarded_torch_policy_path" if decision == "promote" else decision,
        "reason": "; ".join(reasons),
        "hard_gates": {
            "archive": report.get("archive_summary", {}).get("archive"),
            "missing_required": report.get("archive_summary", {}).get("missing_required"),
            "has_pycache": report.get("archive_summary", {}).get("has_pycache"),
            "has_pyc_or_pyo": report.get("archive_summary", {}).get("has_pyc_or_pyo"),
            "games": game_summary.get("games"),
            "errors": game_summary.get("errors"),
            "illegal_action_count": game_summary.get("illegal_action_count"),
            "p99_latency": durability.get("latency_summary", {}).get("p99"),
            "max_latency": durability.get("latency_summary", {}).get("max"),
            "policy_available": policy_stats.get("available"),
            "policy_param_count": policy_stats.get("param_count"),
            "policy_overrides": policy_stats.get("overrides"),
            "policy_eligible": policy_stats.get("eligible"),
        },
        "baseline_comparison": {
            "v0_06d3_official_note": "official Kaggle match completed in about 2 minutes",
            "gameplay_baseline": BASELINE_GAMEPLAY,
            "local_win_rate": game_summary.get("win_rate"),
        },
        "known_risks": [
            "The model imitates the existing rule teacher; this validates plumbing more than final strength.",
            "Random-agent durability is not a Kaggle-strength proof.",
            "Overrides are restricted to single-option low-risk contexts and may be too conservative.",
        ],
        "next_candidates": [
            "Train on replay/log action targets instead of only self-generated teacher labels.",
            "Add a value or outcome target for contexts where teacher imitation cannot improve policy.",
            "Run ablations with guard thresholds and context allow-list while keeping holdout untouched.",
        ],
    }
    write_json(OUTPUT_DIR / f"{RUN_PREFIX}-promotion-decision.json", obj)
    return obj


def main():
    random.seed(RANDOM_SEED)
    os.environ.setdefault("POKEMON_TORCH_THREADS", str(TORCH_THREADS))
    status = base.ensure_cg_importable()
    if not status.get("available"):
        raise RuntimeError(f"cg unavailable: {status}")
    baseline_source, deck = base.extract_baseline()
    data = collect_training_rows(baseline_source, deck)
    training_metrics = train_model(data)
    training_report = {
        "version": RUN_PREFIX,
        "baseline_gameplay": BASELINE_GAMEPLAY,
        "train_games_requested": TRAIN_GAMES,
        "train_epochs": TRAIN_EPOCHS,
        "torch_threads": TORCH_THREADS,
        "data_summary": data["summary"],
        "collection_errors": data["errors"][:20],
        "training_metrics": training_metrics,
    }
    write_json(OUTPUT_DIR / f"{RUN_PREFIX}-training_report.json", training_report)
    policy_source = inject_policy(baseline_source)
    (OUTPUT_DIR / f"{RUN_PREFIX}-main.py").write_text(policy_source, encoding="utf-8")
    archive_summary = base.build_probe_archive(policy_source, deck)
    durability = base.run_durability(policy_source, deck)
    report = {
        "version": RUN_PREFIX,
        "baseline": "pokemon-20260623-v0-06d3 runtime / pokemon-20260623-v0-06d1 gameplay",
        "purpose": "guarded trained PyTorch reranker with real action-change path",
        "training_report": training_report,
        "archive_summary": archive_summary,
        "durability": durability,
        "decision_inputs": {
            "param_count": training_metrics.get("param_count"),
            "valid_top1": training_metrics.get("valid_top1"),
            "errors": durability.get("game_summary", {}).get("errors"),
            "illegal_action_count": durability.get("game_summary", {}).get("illegal_action_count"),
            "p99_latency": durability.get("latency_summary", {}).get("p99"),
            "max_latency": durability.get("latency_summary", {}).get("max"),
            "policy_stats": durability.get("torch_probe_stats", {}),
        },
    }
    write_json(OUTPUT_DIR / f"{RUN_PREFIX}-runtime_report.json", report)
    decision = write_promotion(report)
    print(json.dumps({
        "training": training_report["data_summary"],
        "valid_top1": training_metrics.get("valid_top1"),
        "durability": durability.get("game_summary", {}),
        "latency": durability.get("latency_summary", {}),
        "policy_stats": durability.get("torch_probe_stats", {}),
        "decision": decision.get("decision"),
        "archive": archive_summary.get("archive"),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
