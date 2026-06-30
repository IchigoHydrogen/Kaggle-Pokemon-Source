"""v09d9 prototype — FULL MY-TURN BEAM SEARCH (opponent-agnostic).

v09d4 (1-ply: optimize first action, _base_agent rolls out the rest) = Kaggle 973.5.
Limitation: only the first action is optimized; the rest of my turn is imitation.

This prototype SEARCHES my whole remaining turn (perfect info) with a beam, finds
the action sequence maximizing end-of-turn eval, and plays the FIRST action of the
best line. Opponent is never simulated (2-ply opponent-reading failed: Kaggle 696).

Compare beam_agent vs the v09d4 module in mirror self-play. For opponent-agnostic
changes the mirror is a (conservative) directional proxy: a mirror win strongly
predicts a Kaggle win (v09d4 beat base in mirror and won big on Kaggle).
"""
import sys, time, ctypes, json, importlib.util
sys.path.insert(0, '/kaggle/input/competitions/pokemon-tcg-ai-battle/sample_submission')
sys.path.insert(0, '/tmp/realagent')  # FIRST
import cg.game as game
import cg.api as api
from cg.sim import lib
import main as real   # base agent (rule+LGBM) for non-UNKNOWN_0

my_deck = real.my_deck
BEAM_W = int(sys.argv[2]) if len(sys.argv) > 2 else 6
MAX_DEPTH = int(sys.argv[3]) if len(sys.argv) > 3 else 12
BUDGET = int(sys.argv[4]) if len(sys.argv) > 4 else 600


def load_mod(d, name):
    sys.path.insert(0, d)
    spec = importlib.util.spec_from_file_location(name, d + '/main.py')
    m = importlib.util.module_from_spec(spec); sys.modules[name] = m
    spec.loader.exec_module(m); return m

V4 = load_mod('/tmp/agent_v09d4', 'agent_v09d4')


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
    pls = cur['players']; me, op = pls[p], pls[1 - p]
    my_prize = len(me.get('prize') or []); op_prize = len(op.get('prize') or [])
    oa = (op.get('active') or [None]); op_hp = int(oa[0].get('hp', 0)) if (oa and oa[0]) else 0
    my_hand = me.get('handCount', 0) or len(me.get('hand') or [])
    return 1000.0 * (op_prize - my_prize) - 2.0 * op_hp + (_hp(me) - _hp(op)) + 5.0 * my_hand


def _is_ctx0_node(obs):
    cur = obs.get('current')
    if cur is None or cur.get('result', -1) != -1 or obs.get('select') is None:
        return False
    sel = obs['select']
    n = len(sel.get('option') or []); mn = int(sel.get('minCount', 1) or 1)
    return sel.get('context') == 0 and n >= 2 and mn <= 1


def _fast_forward(state, p, steps):
    """From a node, advance with _base_agent (rule base) through non-ctx0 / opponent-
    irrelevant decisions until we hit a ctx0 branch point, my-turn-end, or terminal."""
    node = state
    for _ in range(80):
        obs = node['observation']; cur = obs.get('current')
        if cur is None or cur.get('result', -1) != -1 or cur.get('yourIndex') != p or obs.get('select') is None:
            return node, True   # terminal-for-eval
        if _is_ctx0_node(obs):
            return node, False  # a ctx0 branch point
        # advance this (non-ctx0) decision with the rule base
        try:
            ch = real.agent(obs)
        except Exception:
            ch = fv(obs)
        if ch is None:
            return node, True
        try:
            nxt = raw_step(node['searchId'], ch); steps[0] += 1
            if nxt.get('error', 0) != 0:
                return node, True
            node = nxt['state']
        except Exception:
            return node, True
    return node, True


def beam_first_action(obs_dict, p, rid):
    """Beam search over the ctx0 decision SEQUENCE only; the rule base (_base_agent)
    fast-forwards everything between ctx0 branch points. This optimizes the decisions
    where we have an edge while keeping the strong rule base in control elsewhere."""
    steps = [0]
    root = {'observation': obs_dict, 'searchId': rid}
    frontier = [(root, None)]          # (ctx0-node, first_opt)
    best_val, best_first = -1e18, 0
    for _ in range(MAX_DEPTH):
        children = []
        for state, fo in frontier:
            obs = state['observation']
            sel = obs['select']; opts = sel.get('option') or []
            for oi in range(len(opts)):
                if steps[0] >= BUDGET:
                    break
                try:
                    child = raw_step(state['searchId'], [oi]); steps[0] += 1
                    if child.get('error', 0) != 0:
                        continue
                except Exception:
                    continue
                ff_node, is_terminal = _fast_forward(child['state'], p, steps)
                first = fo if fo is not None else oi
                if is_terminal:
                    v = ev(ff_node['observation'], p)
                    if v > best_val:
                        best_val, best_first = v, first
                else:
                    children.append((ff_node, first))
        if not children or steps[0] >= BUDGET:
            break
        children.sort(key=lambda sc: ev(sc[0]['observation'], p), reverse=True)
        frontier = children[:BEAM_W]
    for state, fo in frontier:   # any non-terminal leaves left → evaluate as-is
        v = ev(state['observation'], p)
        if v > best_val:
            best_val, best_first = v, (fo if fo is not None else 0)
    return [best_first]


def beam_agent(obs_dict):
    sel = obs_dict.get('select')
    if not sel: return real.agent(obs_dict)
    ctx = sel.get('context'); n = len(sel.get('option') or []); mn = int(sel.get('minCount', 1) or 1)
    if ctx != 0 or n < 2 or mn > 1:
        try: return real.agent(obs_dict)
        except Exception: return fv(obs_dict)
    g = real.__dict__
    snap = (g.get('pre_turn'), g.get('ability_used_dudunsparce'), g.get('ability_used_fezandipiti'))
    try:
        ob = api.to_observation_class(obs_dict); stt = ob.current
        if stt is None or getattr(ob, 'search_begin_input', None) is None:
            return real.agent(obs_dict)
        p = stt.yourIndex; me, opp = stt.players[p], stt.players[1 - p]
        yd = list(my_deck); yp = list(my_deck)[:max(1, len(me.prize))]
        od = list(my_deck); op_ = list(my_deck)[:max(1, len(opp.prize))]
        oh = list(my_deck)[:max(1, opp.handCount)]
        oa = [741] if (len(opp.active) > 0 and opp.active[0] is None) else []
        root = api.search_begin(ob, yd, yp, od, op_, oh, oa)
        res = beam_first_action(obs_dict, p, root.searchId)
        try: api.search_end()
        except Exception: pass
        return res
    except Exception:
        return real.agent(obs_dict)
    finally:
        g['pre_turn'], g['ability_used_dudunsparce'], g['ability_used_fezandipiti'] = snap


def v4_pol(o):
    try: return V4.agent(o)
    except Exception: return fv(o)


def play(p0, p1, max_steps=2000):
    real.pre_turn = -1
    if hasattr(V4, 'pre_turn'): V4.pre_turn = -1
    obs, sd = game.battle_start(my_deck, my_deck)
    if obs is None: return None
    pols = [p0, p1]
    for _ in range(max_steps):
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
    std = (wr * (1 - wr) / tot) ** 0.5 if tot else 0
    print(f'[{name}] A={aw} B={bw} draw={dr} err={er} | wr={wr:.3f} (+-{1.96*std:.3f}) | {time.perf_counter()-t0:.1f}s')
    return wr


if __name__ == '__main__':
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    print(f'=== v09d9 beam (W={BEAM_W} D={MAX_DEPTH} budget={BUDGET}) vs v09d4, N={N} ===')
    match('beam vs v09d4 (>0.50 = beam better)', beam_agent, v4_pol, N)
