"""Test a 1-ply rollout agent that uses the LEARNED value function as the eval,
vs the v09d4 agent (hand-crafted eval) in mirror self-play.

If learned-eval >= v09d4, the learned value is at least not harmful (candidate for
Kaggle). If it loses, the value function is confounded (correlation != action value).
"""
import sys, time, ctypes, json, math, importlib.util
sys.path.insert(0, '/kaggle/input/competitions/pokemon-tcg-ai-battle/sample_submission')
sys.path.insert(0, '/tmp/realagent')  # FIRST
import cg.game as game
import cg.api as api
from cg.sim import lib
import main as real

my_deck = real.my_deck
VM = json.load(open('/kaggle/working/value_model.json'))
FE, MU, SD, CO, IC = VM['feats'], VM['mu'], VM['sd'], VM['coef'], VM['intercept']
print('value model AUC_valid:', VM['auc_valid'])

V4 = (lambda d, n: (lambda s, m: (sys.modules.__setitem__(n, m), s.loader.exec_module(m), m)[-1])(
    importlib.util.spec_from_file_location(n, d + '/main.py'),
    importlib.util.module_from_spec(importlib.util.spec_from_file_location(n, d + '/main.py'))))('/tmp/agent_v09d4', 'av4')


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


def _ids(pl):
    out = []
    for p in (pl.get('active') or []):
        if p: out.append(p.get('id'))
    for p in (pl.get('bench') or []):
        if p: out.append(p.get('id'))
    return out


def learned_eval(obs, p):
    cur = obs.get('current') if obs else None
    if cur is None: return -1e9
    me, op = cur['players'][p], cur['players'][1 - p]
    ids = _ids(me)
    oa = (op.get('active') or [None]); op_hp = int(oa[0].get('hp', 0)) if (oa and oa[0]) else 0
    ma = (me.get('active') or [None]); my_hp = int(ma[0].get('hp', 0)) if (ma and ma[0]) else 0
    my_act_e = len(ma[0].get('energyCards') or []) if (ma and ma[0]) else 0
    f = {
        'prize_diff': len(op.get('prize') or []) - len(me.get('prize') or []),
        'my_prize': len(me.get('prize') or []), 'op_prize': len(op.get('prize') or []),
        'my_active_hp': my_hp, 'op_active_hp': op_hp,
        'my_total_hp': _hp(me), 'op_total_hp': _hp(op),
        'my_hand': me.get('handCount', 0) or len(me.get('hand') or []),
        'op_hand': op.get('handCount', 0) or len(op.get('hand') or []),
        'my_deck': me.get('deckCount', 0), 'op_deck': op.get('deckCount', 0),
        'my_bench': len(me.get('bench') or []), 'op_bench': len(op.get('bench') or []),
        'my_active_energy': my_act_e,
        'my_alakazam': sum(1 for i in ids if i == 743),
        'my_kadabra': sum(1 for i in ids if i == 742),
        'my_abra': sum(1 for i in ids if i == 741),
        'turn': cur.get('turn', 0),
    }
    z = IC
    for i, name in enumerate(FE):
        z += CO[i] * ((f.get(name, 0) - MU[i]) / SD[i])
    return 1.0 / (1.0 + math.exp(-max(-30, min(30, z))))   # P(win)


def learned_agent(obs_dict):
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
        root = api.search_begin(ob, yd, yp, od, op_, oh, oa); rid = root.searchId
        best_i, best_v = 0, -1e18
        for oi in range(n):
            try:
                child = raw_step(rid, [oi])
                if child.get('error', 0) != 0: continue
                # roll out rest of MY turn with rule base, then learned eval
                node = child['state']
                for _ in range(60):
                    o2 = node['observation']; c2 = o2.get('current')
                    if c2 is None or c2.get('result', -1) != -1 or c2.get('yourIndex') != p or o2.get('select') is None:
                        break
                    try: ch = real.agent(o2)
                    except Exception: ch = fv(o2)
                    if ch is None: break
                    nx = raw_step(node['searchId'], ch)
                    if nx.get('error', 0) != 0: break
                    node = nx['state']
                v = learned_eval(node['observation'], p)
            except Exception:
                v = -1e17
            if v > best_v: best_v, best_i = v, oi
        try: api.search_end()
        except Exception: pass
        return [best_i]
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


if __name__ == '__main__':
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    print(f'=== learned-eval rollout vs v09d4, N={N} ===')
    match('learned vs v09d4 (>0.50 = learned better)', learned_agent, v4_pol, N)
