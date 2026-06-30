"""Track B step 3 — forward-sim vs the REAL rule+LGBM agent (Kaggle 792).

Loads the actual v08d34 submission agent (rule base + LGBM reranker, with the
unknown0_lgbm.txt model present) and pits it against the 1-ply forward-sim greedy
agent in engine self-play. This answers the decisive question:
  does forward simulation rival / beat / augment the strong agent?

Matches:
  1. real vs first_valid       (calibrate real agent strength)
  2. fwdsim vs first_valid     (recalibrate forward-sim)
  3. fwdsim vs real            (DECISIVE)
"""
import sys, time

REAL_DIR = '/tmp/realagent'
CG_PARENT = '/kaggle/input/competitions/pokemon-tcg-ai-battle/sample_submission'
sys.path.insert(0, CG_PARENT)    # fallback for `import cg`
sys.path.insert(0, REAL_DIR)     # FIRST: real agent (has main.py + cg/ + lgbm files)

import cg.game as game
import cg.api as api
import main as realmod   # the real submission agent module

print('real agent loaded. LGBM present:', getattr(realmod, '_U0_LGBM', None) is not None)

my_deck = realmod.my_deck


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


def real_policy(obs_dict):
    try:
        return realmod.agent(obs_dict)
    except Exception:
        return first_valid(obs_dict)


def _hp_sum(pstate):
    tot = 0
    if pstate.active:
        for p in pstate.active:
            if p is not None:
                tot += int(getattr(p, 'hp', 0) or 0)
    for p in (pstate.bench or []):
        if p is not None:
            tot += int(getattr(p, 'hp', 0) or 0)
    return tot


def _eval_state(state, p):
    me = state.players[p]
    op = state.players[1 - p]
    my_prize = len(me.prize or [])
    op_prize = len(op.prize or [])
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
        opp_active = [741] if (len(opp.active) > 0 and opp.active[0] is None) else []
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
    # reset real agent per-turn globals
    for gname in ('pre_turn',):
        if hasattr(realmod, gname):
            setattr(realmod, gname, -1)
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
    a_wins = b_wins = draws = errs = 0
    t0 = time.perf_counter()
    for g in range(n_games):
        if g % 2 == 0:
            res, status = play_game(polA, polB); a_is = 0
        else:
            res, status = play_game(polB, polA); a_is = 1
        game.battle_finish()
        if status != 'ok' or res is None:
            errs += 1; continue
        if res == 2: draws += 1
        elif res == a_is: a_wins += 1
        else: b_wins += 1
    dt = time.perf_counter() - t0
    tot = a_wins + b_wins + draws
    wr = a_wins / tot if tot else 0.0
    print(f'[{name}] A_wins={a_wins} B_wins={b_wins} draws={draws} errs={errs} '
          f'| A_winrate={wr:.3f} | {dt:.1f}s ({dt/max(1,n_games):.2f}s/game)')
    return wr


if __name__ == '__main__':
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    print(f'=== Track B v2, N={N} per match ===')
    run_match('real vs first_valid', real_policy, first_valid, N)
    run_match('fwdsim vs first_valid', fwdsim_greedy, first_valid, N)
    run_match('fwdsim vs real (DECISIVE)', fwdsim_greedy, real_policy, N)
