"""v10d2 pre-implementation diagnostic: is there headroom for sim x imitation gating?

On real ctx0 (UNKNOWN_0) decisions during self-play, compute BOTH:
  - sim_values[oi]: v09d4 forward-sim rollout value of each option
  - lgbm_scores[oi]: the imitation LGBM score of each option
Log: how often sim is "flat" (top-vs-2nd margin small => sim can't distinguish),
how often sim's pick != lgbm's pick, and the option count. If sim is flat often
AND disagrees with lgbm, the tiebreak/gate has real headroom.
"""
import sys, ctypes, json, importlib.util
sys.path.insert(0, '/kaggle/input/competitions/pokemon-tcg-ai-battle/sample_submission')
sys.path.insert(0, '/tmp/realagent')  # FIRST
import cg.game as game
import cg.api as api
from cg.sim import lib
import main as real

my_deck = real.my_deck


def fv(o):
    sel = o.get('select')
    if sel is None: return None
    mn = int(sel.get('minCount', 1) or 1); n = len(sel.get('option', []) or [])
    return list(range(min(max(1, mn), n))) or [0]


def raw_step(sid, sel):
    arr = (ctypes.c_int * len(sel))(*sel)
    bs = lib.SearchStep(api.agent_ptr, sid, arr, len(sel))
    return json.loads(bs.decode('utf-8') if isinstance(bs, (bytes, bytearray)) else bs)


def _hp(pl):
    t = 0
    for p in (pl.get('active') or []):
        if p: t += int(p.get('hp', 0) or 0)
    for p in (pl.get('bench') or []):
        if p: t += int(p.get('hp', 0) or 0)
    return t


def ev(obs, p):
    cur = obs.get('current') if obs else None
    if cur is None: return -1e9
    me, op = cur['players'][p], cur['players'][1 - p]
    my_prize = len(me.get('prize') or []); op_prize = len(op.get('prize') or [])
    oa = (op.get('active') or [None]); op_hp = int(oa[0].get('hp', 0)) if (oa and oa[0]) else 0
    my_hand = me.get('handCount', 0) or len(me.get('hand') or [])
    return 1000.0 * (op_prize - my_prize) - 2.0 * op_hp + (_hp(me) - _hp(op)) + 5.0 * my_hand


def rollout_value(child, p, cap=60):
    node = child
    for _ in range(cap):
        st = node.get('state', node); obs = st.get('observation'); sid = st.get('searchId')
        cur = obs.get('current') if obs else None
        if cur is None or cur.get('result', -1) != -1 or cur.get('yourIndex') != p:
            return ev(obs, p)
        if obs.get('select') is None: return ev(obs, p)
        try: ch = real.agent(obs)
        except Exception: return ev(obs, p)
        if ch is None: return ev(obs, p)
        try:
            node = raw_step(sid, ch)
            if node.get('error', 0) != 0: return ev(obs, p)
        except Exception: return ev(obs, p)
    return ev(node.get('state', node).get('observation'), p)


def sim_values_and_lgbm(obs_dict):
    """Return (sim_values, lgbm_scores) for a ctx0 decision, or None."""
    sel = obs_dict.get('select')
    n = len(sel.get('option') or [])
    ob = api.to_observation_class(obs_dict); stt = ob.current
    if stt is None or getattr(ob, 'search_begin_input', None) is None:
        return None
    p = stt.yourIndex; me, opp = stt.players[p], stt.players[1 - p]
    yd = list(my_deck); yp = list(my_deck)[:max(1, len(me.prize))]
    od = list(my_deck); op_ = list(my_deck)[:max(1, len(opp.prize))]
    oh = list(my_deck)[:max(1, opp.handCount)]
    oa = [741] if (len(opp.active) > 0 and opp.active[0] is None) else []
    try:
        root = api.search_begin(ob, yd, yp, od, op_, oh, oa); rid = root.searchId
    except Exception:
        return None
    sims = []
    for oi in range(n):
        try:
            child = raw_step(rid, [oi])
            sims.append(rollout_value(child, p) if child.get('error', 0) == 0 else -1e17)
        except Exception:
            sims.append(-1e17)
    try: api.search_end()
    except Exception: pass
    try:
        lg = real._u0_lgbm_scores(ob, ob.select, p, 1 - p)
    except Exception:
        lg = None
    return sims, lg


import statistics
def run(N):
    real.pre_turn = -1
    stats = {'ctx0': 0, 'flat': 0, 'disagree': 0, 'flat_and_disagree': 0,
             'lg_none': 0, 'nopt_sum': 0}
    margins = []
    for g in range(N):
        real.pre_turn = -1
        obs, sd = game.battle_start(my_deck, my_deck)
        if obs is None: continue
        for _ in range(2000):
            cur = obs.get('current')
            if cur is not None and cur.get('result', -1) != -1: break
            sel = obs.get('select')
            if sel is None: break
            ctx = sel.get('context'); n = len(sel.get('option') or []); mn = int(sel.get('minCount', 1) or 1)
            if ctx == 0 and n >= 2 and mn <= 1:
                res = sim_values_and_lgbm(obs)
                if res is not None:
                    sims, lg = res
                    stats['ctx0'] += 1; stats['nopt_sum'] += n
                    order = sorted(range(n), key=lambda i: -sims[i])
                    top, sec = sims[order[0]], sims[order[1]]
                    margin = top - sec
                    margins.append(margin)
                    is_flat = margin < 1.0   # near-tied sim values (no prize/KO distinction)
                    if is_flat: stats['flat'] += 1
                    sim_pick = order[0]
                    if lg is None or len(lg) != n:
                        stats['lg_none'] += 1; lg_pick = None
                    else:
                        lg_pick = max(range(n), key=lambda i: lg[i])
                    disagree = (lg_pick is not None and lg_pick != sim_pick)
                    if disagree: stats['disagree'] += 1
                    if is_flat and disagree: stats['flat_and_disagree'] += 1
                # advance with sim argmax (v09d4 behavior)
                ch = real.agent(obs)
            else:
                ch = real.agent(obs)
            if ch is None: ch = fv(obs)
            try: obs = game.battle_select(ch)
            except Exception: break
        game.battle_finish()
    c = max(1, stats['ctx0'])
    print('=== v10d2 gating headroom diagnostic ===')
    print(f"ctx0 decisions sampled: {stats['ctx0']}, avg n_options={stats['nopt_sum']/c:.1f}")
    print(f"sim FLAT (top-2nd margin<1.0): {stats['flat']} ({100*stats['flat']/c:.1f}%)")
    print(f"sim-pick != lgbm-pick (disagree): {stats['disagree']} ({100*stats['disagree']/c:.1f}%)")
    print(f"FLAT and DISAGREE (gate headroom): {stats['flat_and_disagree']} ({100*stats['flat_and_disagree']/c:.1f}%)")
    print(f"lgbm unavailable: {stats['lg_none']}")
    if margins:
        margins.sort()
        print(f"sim margin pctiles: p10={margins[len(margins)//10]:.1f} p50={margins[len(margins)//2]:.1f} p90={margins[9*len(margins)//10]:.1f}")


if __name__ == '__main__':
    run(int(sys.argv[1]) if len(sys.argv) > 1 else 15)
