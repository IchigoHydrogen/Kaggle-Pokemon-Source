"""Track B step 2 — self-play win-rate harness + 1-ply forward-sim greedy agent.

This upgrades our north-star metric from imitation top1 to ACTUAL WIN RATE,
measured inside the competition engine (which runs ~1ms/decision).

Agents:
  - first_valid: pick the minimal valid selection (baseline)
  - fwdsim_greedy: for any select with >=2 single-pick options, forward-simulate
      each option via search_begin/search_step, evaluate the resulting board from
      the mover's perspective, pick the best. Falls back to first_valid otherwise.

Eval(state, p) = 1000*(op_prize_left - my_prize_left) + (sum_my_hp - sum_op_hp)
  (lower my prizes remaining = winning; healthier board = better)

We run N games fwdsim(p0) vs first_valid(p1) and N games with sides swapped,
report win rate. If fwdsim beats first_valid clearly, forward-sim has real value
and is worth integrating into main.py for UNKNOWN_0.
"""
import sys, time

CG_PARENT = '/kaggle/input/competitions/pokemon-tcg-ai-battle/sample_submission'
sys.path.insert(0, CG_PARENT)

my_deck = [741, 741, 741, 741, 742, 742, 742, 742, 743, 743, 743, 305, 305, 305, 66, 66, 66, 140, 1231, 1231, 1231, 1231, 1225, 1225, 1225, 1225, 1182, 1182, 1182, 1184, 1184, 1086, 1086, 1086, 1086, 1152, 1152, 1152, 1152, 1079, 1079, 1079, 1081, 1081, 1081, 1081, 1129, 1097, 1156, 1174, 1266, 1266, 1266, 19, 19, 19, 19, 13, 5, 5]

import cg.game as game
import cg.api as api


def _sel_meta(obs_dict):
    sel = obs_dict.get('select')
    if sel is None:
        return None, 0, 1
    opts = sel.get('option', []) or []
    mn = int(sel.get('minCount', 1) or 1)
    return sel, len(opts), mn


def first_valid(obs_dict):
    sel, n, mn = _sel_meta(obs_dict)
    if sel is None:
        return None
    k = max(1, mn)
    idxs = list(range(min(k, n)))
    return idxs if idxs else [0]


def _hp_sum(pstate):
    tot = 0
    for grp in ([pstate.active] if pstate.active else []):
        for p in grp:
            if p is not None:
                tot += int(getattr(p, 'hp', 0) or 0)
    for p in (pstate.bench or []):
        if p is not None:
            tot += int(getattr(p, 'hp', 0) or 0)
    return tot


def _eval_state(state, p):
    """Value of a resulting State from player p's perspective."""
    me = state.players[p]
    op = state.players[1 - p]
    my_prize = len([c for c in (me.prize or []) if True])   # remaining prize slots
    op_prize = len([c for c in (op.prize or []) if True])
    return 1000.0 * (op_prize - my_prize) + (_hp_sum(me) - _hp_sum(op))


def fwdsim_greedy(obs_dict):
    sel, n, mn = _sel_meta(obs_dict)
    if sel is None:
        return None
    if n < 2 or mn > 1:
        return first_valid(obs_dict)
    try:
        ob = api.to_observation_class(obs_dict)
        st = ob.current
        if st is None or getattr(ob, 'search_begin_input', None) is None:
            return first_valid(obs_dict)
        p = st.yourIndex
        opp = st.players[1 - p]
        me = st.players[p]
        your_deck = list(my_deck)
        your_prize = list(my_deck)[:max(1, len(me.prize))]
        opp_deck = list(my_deck)
        opp_prize = list(my_deck)[:max(1, len(opp.prize))]
        opp_hand = list(my_deck)[:max(1, opp.handCount)]
        opp_active = []
        if len(opp.active) > 0 and opp.active[0] is None:
            opp_active = [741]
        root = api.search_begin(ob, your_deck, your_prize, opp_deck,
                                opp_prize, opp_hand, opp_active)
        rid = root.searchId
        best_i, best_v = 0, -1e18
        for oi in range(n):
            try:
                nxt = api.search_step(rid, [oi])
                ns = nxt.observation.current
                v = _eval_state(ns, p) if ns is not None else -1e17
            except Exception:
                v = -1e17
            if v > best_v:
                best_v, best_i = v, oi
        api.search_end()
        return [best_i]
    except Exception:
        return first_valid(obs_dict)


def play_game(policy0, policy1, max_steps=2000):
    policies = [policy0, policy1]
    obs, sd = game.battle_start(my_deck, my_deck)
    if obs is None:
        return None, 'start_fail'
    for _ in range(max_steps):
        cur = obs.get('current')
        if cur is not None and cur.get('result', -1) != -1:
            return cur['result'], 'ok'
        sel = obs.get('select')
        if sel is None:
            return None, 'no_select'
        p = cur['yourIndex'] if cur else 0
        choice = policies[p](obs)
        if choice is None:
            return None, 'no_choice'
        try:
            obs = game.battle_select(choice)
        except Exception as e:
            return None, f'select_err:{e!r}'
    return None, 'max_steps'


def run_match(name, polA, polB, n_games):
    """polA as player0 for half, player1 for half. Returns A win rate."""
    a_wins = b_wins = draws = errs = 0
    t0 = time.perf_counter()
    for g in range(n_games):
        if g % 2 == 0:
            res, status = play_game(polA, polB)
            a_is = 0
        else:
            res, status = play_game(polB, polA)
            a_is = 1
        game.battle_finish()
        if status != 'ok' or res is None:
            errs += 1
            continue
        if res == 2:
            draws += 1
        elif res == a_is:
            a_wins += 1
        else:
            b_wins += 1
    dt = time.perf_counter() - t0
    tot = a_wins + b_wins + draws
    wr = a_wins / tot if tot else 0.0
    print(f'[{name}] A_wins={a_wins} B_wins={b_wins} draws={draws} errs={errs} '
          f'| A_winrate={wr:.3f} | {dt:.1f}s ({dt/max(1,n_games):.2f}s/game)')
    return wr


if __name__ == '__main__':
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    print(f'=== self-play harness, N={N} per match ===')
    # sanity: first_valid mirror should be ~50%
    run_match('sanity firstvalid vs firstvalid', first_valid, first_valid, N)
    # the real test
    run_match('fwdsim_greedy vs first_valid', fwdsim_greedy, first_valid, N)
