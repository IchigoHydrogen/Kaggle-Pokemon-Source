"""Validation-unlock feasibility: a DECK-AGNOSTIC strong forward-sim agent.

v09d4's strength uses the Alakazam rule base (deck-specific). To validate opponent-
aware features we need a STRONG opponent on OTHER decks. Test whether a pure
forward-sim agent (no rule base, no LGBM) can be competitive: for every decision,
search each option and roll out the rest of MY turn with a GREEDY-EVAL policy
(each rollout step picks the option maximizing the prize eval), evaluate end-of-turn,
pick the best first option. If this is competitive vs v09d4 on Alakazam, it can serve
as a strong deck-agnostic opponent on Hop/Lucario decks -> validation unlock.
"""
import sys, time, ctypes, json, importlib.util
sys.path.insert(0, '/kaggle/input/competitions/pokemon-tcg-ai-battle/sample_submission')
sys.path.insert(0, '/tmp/agent_v09d4')
import cg.game as game
import cg.api as api
from cg.sim import lib

spec = importlib.util.spec_from_file_location('v4', '/tmp/agent_v09d4/main.py')
V4 = importlib.util.module_from_spec(spec); sys.modules['v4'] = V4; spec.loader.exec_module(V4)
ALAK = V4.my_deck


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
    mp = len(me.get('prize') or []); opp = len(op.get('prize') or [])
    oa = (op.get('active') or [None]); op_hp = int(oa[0].get('hp', 0)) if (oa and oa[0]) else 0
    mh = me.get('handCount', 0) or len(me.get('hand') or [])
    return 1000.0 * (opp - mp) - 2.0 * op_hp + (_hp(me) - _hp(op)) + 5.0 * mh

def ev_leaf(obs, p):
    cur = obs.get('current') if obs else None
    if cur is not None:
        _r=cur.get('result',-1)
        if _r==p: return 1e7
        if _r==(1-p): return -1e7
        if _r==2: return -1e5
    return ev(obs, p)


def greedy_rollout(state, p, deck, cap=60):
    """Roll out p's turn with a 1-step greedy-eval policy; return end-of-turn obs."""
    node = state
    for _ in range(cap):
        st = node.get('state', node); obs = st.get('observation'); sid = st.get('searchId')
        cur = obs.get('current') if obs else None
        if cur is None or cur.get('result', -1) != -1 or cur.get('yourIndex') != p:
            return obs
        sel = obs.get('select')
        if sel is None: return obs
        nn = len(sel.get('option') or []); mn = int(sel.get('minCount', 1) or 1)
        if nn == 0: return obs
        # 1-step greedy among options (single-pick); else first-valid
        best_oi, best_v, best_child = 0, -1e18, None
        if mn <= 1:
            for oi in range(nn):
                try:
                    ch = raw_step(sid, [oi])
                    if ch.get('error', 0) != 0: continue
                    v = ev(ch['state']['observation'], p)
                    if v > best_v: best_v, best_oi, best_child = v, oi, ch
                except Exception:
                    continue
            if best_child is None: return obs
            node = best_child
        else:
            try:
                node = raw_step(sid, list(range(min(mn, nn))))
                if node.get('error', 0) != 0: return obs
            except Exception:
                return obs
    return node.get('state', node).get('observation')


def generic_fwdsim(obs_dict, deck):
    sel = obs_dict.get('select')
    if sel is None: return None
    n = len(sel.get('option') or []); mn = int(sel.get('minCount', 1) or 1)
    if n < 2 or mn > 1:
        return fv(obs_dict)
    try:
        ob = api.to_observation_class(obs_dict); stt = ob.current
        if stt is None or getattr(ob, 'search_begin_input', None) is None:
            return fv(obs_dict)
        p = stt.yourIndex; me, opp = stt.players[p], stt.players[1 - p]
        yd = list(deck); yp = list(deck)[:max(1, len(me.prize))]
        od = list(deck); op_ = list(deck)[:max(1, len(opp.prize))]
        oh = list(deck)[:max(1, opp.handCount)]
        oa = [deck[0]] if (len(opp.active) > 0 and opp.active[0] is None) else []
        root = api.search_begin(ob, yd, yp, od, op_, oh, oa); rid = root.searchId
        best_i, best_v = 0, -1e18
        for oi in range(n):
            try:
                child = raw_step(rid, [oi])
                end = greedy_rollout(child['state'], p, deck) if child.get('error', 0) == 0 else None
                v = ev_leaf(end, p) if end else -1e17
            except Exception:
                v = -1e17
            if v > best_v: best_v, best_i = v, oi
        try: api.search_end()
        except Exception: pass
        return [best_i]
    except Exception:
        return fv(obs_dict)


def v4_pol(o):
    try: return V4.agent(o)
    except Exception: return fv(o)


def gfs_pol(o):
    return generic_fwdsim(o, ALAK)


def play(p0, p1):
    if hasattr(V4, 'pre_turn'): V4.pre_turn = -1
    obs, sd = game.battle_start(ALAK, ALAK)
    if obs is None: return None
    pols = [p0, p1]
    for _ in range(2000):
        cur = obs.get('current')
        if cur is not None and cur.get('result', -1) != -1: return cur['result']
        if obs.get('select') is None: return None
        pp = cur['yourIndex'] if cur else 0
        ch = pols[pp](obs)
        if ch is None: return None
        try: obs = game.battle_select(ch)
        except Exception: return None
    return None


def match(name, A, B, N):
    aw = bw = dr = er = 0; t0 = time.perf_counter()
    for g in range(N):
        if g % 2 == 0: r = play(A, B); ais = 0
        else: r = play(B, A); ais = 1
        game.battle_finish()
        if r is None: er += 1; continue
        if r == 2: dr += 1
        elif r == ais: aw += 1
        else: bw += 1
    tot = aw + bw + dr; wr = aw / tot if tot else 0
    print(f'[{name}] A={aw} B={bw} draw={dr} err={er} | A_wr={wr:.3f} (+-{1.96*(wr*(1-wr)/max(1,tot))**0.5:.3f}) | {time.perf_counter()-t0:.1f}s')


if __name__ == '__main__':
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    print(f'=== generic_fwdsim (deck-agnostic, no rule base) vs v09d4 on Alakazam, N={N} ===')
    match('generic_fwdsim vs v09d4', gfs_pol, v4_pol, N)
