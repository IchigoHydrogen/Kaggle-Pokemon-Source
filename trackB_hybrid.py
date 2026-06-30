"""Track B step 4 — HYBRID: real agent + forward-sim 'don't miss a prize' safety net.

Calibration showed forward-sim ALONE loses to the real agent (0.133). So instead
of replacing, we AUGMENT: the real agent decides normally, but we forward-simulate
each option and if some option TAKES A PRIZE (my prize_left decreases, i.e. a KO)
that the real agent's choice does NOT, we override to take it. This can only ever
add tactical prize-taking the imitation agent missed.

Measure: hybrid vs real. >0.50 means the safety net helps.
"""
import sys
sys.path.insert(0, '/tmp/realagent')
sys.path.insert(0, '/kaggle/input/competitions/pokemon-tcg-ai-battle/sample_submission')
import trackB_selfplay_v2 as h   # reuse engine, real agent, fwdsim, harness

api = h.api
my_deck = h.my_deck


def hybrid_policy(obs_dict):
    rc = h.real_policy(obs_dict)
    sel, n, mn = h._sel_meta(obs_dict)
    if sel is None or n < 2 or mn > 1:
        return rc
    try:
        ob = api.to_observation_class(obs_dict)
        st = ob.current
        if st is None or getattr(ob, 'search_begin_input', None) is None:
            return rc
        p = st.yourIndex
        me = st.players[p]; opp = st.players[1 - p]
        before_myprize = len(me.prize or [])
        your_deck = list(my_deck)
        your_prize = list(my_deck)[:max(1, len(me.prize))]
        opp_deck = list(my_deck)
        opp_prize = list(my_deck)[:max(1, len(opp.prize))]
        opp_hand = list(my_deck)[:max(1, opp.handCount)]
        opp_active = [741] if (len(opp.active) > 0 and opp.active[0] is None) else []
        root = api.search_begin(ob, your_deck, your_prize, opp_deck,
                                opp_prize, opp_hand, opp_active)
        rid = root.searchId
        rc_idx = rc[0] if rc else 0
        rc_takes = False
        best_prize_i, best_v = None, -1e18
        for oi in range(n):
            try:
                nxt = api.search_step(rid, [oi])
                ns = nxt.observation.current
                if ns is None:
                    continue
                after_myprize = len(ns.players[p].prize or [])
                took = after_myprize < before_myprize
                if took:
                    v = h._eval_state(ns, p)
                    if oi == rc_idx:
                        rc_takes = True
                    if v > best_v:
                        best_v, best_prize_i = v, oi
            except Exception:
                continue
        api.search_end()
        if best_prize_i is not None and not rc_takes:
            return [best_prize_i]   # take the prize the real agent missed
    except Exception:
        pass
    return rc


if __name__ == '__main__':
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    print(f'=== Track B hybrid, N={N} ===')
    h.run_match('hybrid vs real (>0.50 = helps)', hybrid_policy, h.real_policy, N)
    h.run_match('hybrid vs first_valid', hybrid_policy, h.first_valid, N)
