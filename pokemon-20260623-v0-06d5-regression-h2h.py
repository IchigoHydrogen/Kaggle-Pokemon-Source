#!/usr/bin/env python3
"""pokemon-20260623-v0-06d5 regression-localization head-to-head diagnostic.

Hypothesis B: the v0-06d1 phase-aware tactical scoring layer is a strength
regression versus v0-05d6. Decks are byte-identical across versions, so a
same-deck, seat-balanced head-to-head isolates the tactical layer.

Diagnostics only: no agent/deck/model edits, no submission build.
The C engine RNG is not seedable from Python, so reproducibility is
statistical (N + seat balance + Wilson CI), not exact.
"""
import importlib.util
import json
import math
import os
import random
import sys
import tarfile
import time

RUN_PREFIX = "pokemon-20260623-v0-06d5"
WORKING = "/kaggle/working"
OUTPUT_DIR = os.path.join(WORKING, RUN_PREFIX)
CG_PARENT = "/kaggle/input/competitions/pokemon-tcg-ai-battle/sample_submission"
TMP_DIR = os.path.join(OUTPUT_DIR, "agents")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TMP_DIR, exist_ok=True)
sys.path.insert(0, CG_PARENT)

import cg.api as cgapi  # noqa: E402
from cg.api import to_observation_class  # noqa: E402
from cg.game import battle_finish, battle_select, battle_start  # noqa: E402

MAX_STEPS = int(os.environ.get("V06D5_MAX_STEPS", "400"))
H2H_GAMES = int(os.environ.get("V06D5_H2H_GAMES", "240"))      # seat-balanced
SAT_GAMES = int(os.environ.get("V06D5_SAT_GAMES", "80"))       # per weak opponent
SEED = int(os.environ.get("V06D5_SEED", "20260623"))
INCLUDE_D4 = os.environ.get("V06D5_INCLUDE_D4", "1") == "1"


def load_agent(version, label):
    tarp = os.path.join(WORKING, f"{version}-submission.tar.gz")
    with tarfile.open(tarp) as t:
        src = t.extractfile("main.py").read().decode()
        deck = [int(x) for x in t.extractfile("deck.csv").read().decode().split() if x.strip()]
    p = os.path.join(TMP_DIR, f"agent_{label}.py")
    with open(p, "w") as f:
        f.write(src)
    spec = importlib.util.spec_from_file_location(f"v06d5_agent_{label}", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    assert len(deck) == 60, f"{version} deck not 60: {len(deck)}"
    return m, deck


def make_random_agent(deck):
    def random_agent(obs_dict):
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


def make_first_valid_agent(deck):
    def first_valid_agent(obs_dict):
        obs = to_observation_class(obs_dict)
        if obs.select is None:
            return deck
        min_count = max(0, int(obs.select.minCount))
        max_count = min(len(obs.select.option), max(0, int(obs.select.maxCount)))
        k = min(max_count, max(min_count, 0))
        return list(range(k))
    return first_valid_agent


def play_game(agent0, deck0, agent1, deck1, max_steps=MAX_STEPS):
    """Return dict with result (0=>seat0 won, 1=>seat1 won, None=>no winner)."""
    obs, sd = battle_start(deck0, deck1)
    if obs is None:
        return {"result": None, "steps": 0,
                "error": f"battle_start_failed:{getattr(sd,'errorPlayer',None)}:{getattr(sd,'errorType',None)}",
                "illegal_action": False}
    try:
        for step in range(max_steps + 1):
            obc = to_observation_class(obs)
            if obc.current.result >= 0:
                return {"result": int(obc.current.result), "steps": step, "error": "", "illegal_action": False}
            seat = int(obc.current.yourIndex)
            action = agent0(obs) if seat == 0 else agent1(obs)
            if not isinstance(action, list) or any(not isinstance(x, int) for x in action):
                return {"result": None, "steps": step, "error": f"illegal_action_type:{action!r}", "illegal_action": True}
            if obc.select is not None:
                mn = int(obc.select.minCount)
                mx = int(obc.select.maxCount)
                nopt = len(obc.select.option)
                if (len(action) < mn or len(action) > mx or len(set(action)) != len(action)
                        or any((x < 0 or x >= nopt) for x in action)):
                    return {"result": None, "steps": step,
                            "error": f"illegal_action_value:{action!r}:min={mn}:max={mx}:nopt={nopt}",
                            "illegal_action": True}
            obs = battle_select(action)
        return {"result": None, "steps": max_steps, "error": "max_steps_exceeded", "illegal_action": False}
    except Exception as exc:  # noqa: BLE001
        return {"result": None, "steps": None, "error": repr(exc), "illegal_action": False}
    finally:
        try:
            battle_finish()
        except Exception:  # noqa: BLE001
            pass


def wilson_ci(wins, n, z=1.96):
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = wins / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    margin = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (p, max(0.0, center - margin), min(1.0, center + margin))


def head_to_head(name_a, agent_a, deck_a, name_b, agent_b, deck_b, n_games):
    """A vs B, seat-balanced. 'win' counted for A. Draws (None) excluded from win rate."""
    rows = []
    a_wins = 0
    b_wins = 0
    draws = 0
    errors = 0
    illegal = 0
    for i in range(n_games):
        if i % 2 == 0:
            res = play_game(agent_a, deck_a, agent_b, deck_b)
            a_won = res["result"] == 0
            b_won = res["result"] == 1
            seat_a = 0
        else:
            res = play_game(agent_b, deck_b, agent_a, deck_a)
            a_won = res["result"] == 1
            b_won = res["result"] == 0
            seat_a = 1
        if res["error"]:
            errors += 1
        if res["illegal_action"]:
            illegal += 1
        if res["result"] is None:
            draws += 1
        elif a_won:
            a_wins += 1
        elif b_won:
            b_wins += 1
        rows.append({"pair": f"{name_a}_vs_{name_b}", "game": i, "seat_a": seat_a,
                     "result": res["result"], "a_won": bool(a_won), "b_won": bool(b_won),
                     "steps": res["steps"], "error": res["error"], "illegal_action": res["illegal_action"]})
    decided = a_wins + b_wins
    p, lo, hi = wilson_ci(a_wins, decided)
    summary = {
        "pair": f"{name_a}_vs_{name_b}", "n_games": n_games, "decided": decided,
        "a_name": name_a, "b_name": name_b,
        "a_wins": a_wins, "b_wins": b_wins, "draws": draws,
        "errors": errors, "illegal_action_count": illegal,
        "a_win_rate_decided": p, "a_win_rate_wilson_lo": lo, "a_win_rate_wilson_hi": hi,
        "a_seat0_games": (n_games + 1) // 2, "a_seat1_games": n_games // 2,
    }
    return rows, summary


def saturation(name, agent, deck, n_games):
    """agent vs random and first_valid, seat-balanced. Win counted for agent."""
    out = {}
    rows_all = []
    for opp_name, opp in [("random_same_deck", make_random_agent(deck)),
                          ("first_valid_same_deck", make_first_valid_agent(deck))]:
        wins = 0
        decided = 0
        errors = 0
        illegal = 0
        for i in range(n_games):
            if i % 2 == 0:
                res = play_game(agent, deck, opp, deck)
                won = res["result"] == 0
            else:
                res = play_game(opp, deck, agent, deck)
                won = res["result"] == 1
            if res["error"]:
                errors += 1
            if res["illegal_action"]:
                illegal += 1
            if res["result"] is not None:
                decided += 1
                if won:
                    wins += 1
            rows_all.append({"agent": name, "opponent": opp_name, "game": i,
                             "won": bool(won), "result": res["result"], "error": res["error"]})
        p, lo, hi = wilson_ci(wins, decided)
        out[opp_name] = {"agent": name, "opponent": opp_name, "n_games": n_games,
                         "decided": decided, "wins": wins, "win_rate": p,
                         "wilson_lo": lo, "wilson_hi": hi, "errors": errors,
                         "illegal_action_count": illegal}
    return rows_all, out


def main():
    random.seed(SEED)
    t0 = time.time()
    print(f"[v06d5] loading agents; H2H_GAMES={H2H_GAMES} SAT_GAMES={SAT_GAMES} seed={SEED}")
    m5, d5 = load_agent("pokemon-20260623-v0-05d6", "v05d6")
    m6, d6 = load_agent("pokemon-20260623-v0-06d1", "v06d1")
    agents = {"v05d6": (m5.agent, d5), "v06d1": (m6.agent, d6)}
    deck_identical_5_6 = d5 == d6
    if INCLUDE_D4:
        m4, d4 = load_agent("pokemon-20260623-v0-06d4", "v06d4")
        agents["v06d4"] = (m4.agent, d4)
        deck_identical_6_4 = d6 == d4
    else:
        deck_identical_6_4 = None
    print(f"[v06d5] deck identical 05d6==06d1: {deck_identical_5_6}; 06d1==06d4: {deck_identical_6_4}")

    all_rows = []
    h2h_summaries = {}

    # Primary head-to-head: 05d6 vs 06d1 (isolates tactical layer).
    print("[v06d5] head-to-head 05d6 vs 06d1 ...")
    rows, summ = head_to_head("v05d6", m5.agent, d5, "v06d1", m6.agent, d6, H2H_GAMES)
    all_rows.extend(rows)
    h2h_summaries["v05d6_vs_v06d1"] = summ
    print("   ", json.dumps({k: summ[k] for k in ("a_wins", "b_wins", "draws", "a_win_rate_decided",
                                                  "a_win_rate_wilson_lo", "a_win_rate_wilson_hi", "errors")}))

    # Secondary head-to-head: 06d4 (reranker) vs 06d1 (its own teacher).
    if INCLUDE_D4:
        print("[v06d5] head-to-head 06d4 vs 06d1 ...")
        rows, summ = head_to_head("v06d4", agents["v06d4"][0], agents["v06d4"][1],
                                  "v06d1", m6.agent, d6, H2H_GAMES)
        all_rows.extend(rows)
        h2h_summaries["v06d4_vs_v06d1"] = summ
        print("   ", json.dumps({k: summ[k] for k in ("a_wins", "b_wins", "draws", "a_win_rate_decided",
                                                      "a_win_rate_wilson_lo", "a_win_rate_wilson_hi", "errors")}))
        # And 06d4 vs 05d6 (does reranker recover the gap?).
        print("[v06d5] head-to-head 06d4 vs 05d6 ...")
        rows, summ = head_to_head("v06d4", agents["v06d4"][0], agents["v06d4"][1],
                                  "v05d6", m5.agent, d5, H2H_GAMES)
        all_rows.extend(rows)
        h2h_summaries["v06d4_vs_v05d6"] = summ
        print("   ", json.dumps({k: summ[k] for k in ("a_wins", "b_wins", "draws", "a_win_rate_decided",
                                                      "a_win_rate_wilson_lo", "a_win_rate_wilson_hi", "errors")}))

    # Saturation against weak opponents (the suspected non-discriminating axis).
    sat_summaries = {}
    sat_rows = []
    for name in ("v05d6", "v06d1"):
        print(f"[v06d5] saturation {name} vs weak opponents ...")
        r, out = saturation(name, agents[name][0], agents[name][1], SAT_GAMES)
        sat_rows.extend(r)
        sat_summaries[name] = out
        for opp, s in out.items():
            print(f"    {name} vs {opp}: win_rate={s['win_rate']:.3f} "
                  f"[{s['wilson_lo']:.3f},{s['wilson_hi']:.3f}] errors={s['errors']}")

    elapsed = time.time() - t0
    result = {
        "run_prefix": RUN_PREFIX,
        "config": {"h2h_games": H2H_GAMES, "sat_games": SAT_GAMES, "max_steps": MAX_STEPS,
                   "seed": SEED, "include_d4": INCLUDE_D4},
        "deck_identical_05d6_vs_06d1": deck_identical_5_6,
        "deck_identical_06d1_vs_06d4": deck_identical_6_4,
        "engine_reproducibility": "non_deterministic_engine_rng_not_python_seedable",
        "head_to_head": h2h_summaries,
        "saturation": sat_summaries,
        "elapsed_seconds": elapsed,
    }
    with open(os.path.join(OUTPUT_DIR, f"{RUN_PREFIX}-h2h_summary.json"), "w") as f:
        json.dump(result, f, indent=2)
    with open(os.path.join(OUTPUT_DIR, f"{RUN_PREFIX}-h2h_rows.json"), "w") as f:
        json.dump({"h2h_rows": all_rows, "saturation_rows": sat_rows}, f)
    print(f"[v06d5] done in {elapsed:.1f}s -> {OUTPUT_DIR}/{RUN_PREFIX}-h2h_summary.json")
    return result


if __name__ == "__main__":
    main()
