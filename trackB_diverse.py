"""v10d foundation: DIVERSE (non-mirror) opponent validation.

My Alakazam agent vs a deck-agnostic forward-sim-greedy opponent playing a
reconstructed archetype deck (e.g. Hop_Trevenant). Crucially we test whether this
harness REPRODUCES the Kaggle ranking v09d4 > v09d6 (mirror INVERTED it). If it
does, it's a trustworthy detector of opponent-model failures for developing v10d.

Usage: python trackB_diverse.py <my_agent_dir> <opp_deck_json> <N>
"""
import sys, time, ctypes, json, importlib.util
sys.path.insert(0, '/kaggle/input/competitions/pokemon-tcg-ai-battle/sample_submission')
import cg.game as game
import cg.api as api
from cg.sim import lib

MY_DIR = sys.argv[1] if len(sys.argv) > 1 else '/tmp/agent_v09d4'
OPP_DECK_JSON = sys.argv[2] if len(sys.argv) > 2 else '/kaggle/working/opp_deck_Hop_Trevenant.json'
N = int(sys.argv[3]) if len(sys.argv) > 3 else 200

ALAK = [741,741,741,741,742,742,742,742,743,743,743,305,305,305,66,66,66,140,1231,1231,1231,1231,1225,1225,1225,1225,1182,1182,1182,1184,1184,1086,1086,1086,1086,1152,1152,1152,1152,1079,1079,1079,1081,1081,1081,1081,1129,1097,1156,1174,1266,1266,1266,19,19,19,19,13,5,5]
OPP_DECK = json.load(open(OPP_DECK_JSON))['deck']

spec = importlib.util.spec_from_file_location('myag', MY_DIR + '/main.py')
MY = importlib.util.module_from_spec(spec); sys.modules['myag'] = MY; spec.loader.exec_module(MY)
print('my agent:', MY_DIR, 'LGBM:', getattr(MY, '_U0_LGBM', None) is not None, '| opp deck:', OPP_DECK_JSON)


def fv(o):
    sel = o.get('select')
    if sel is None: return None
    mn = int(sel.get('minCount', 1) or 1); n = len(sel.get('option', []) or [])
    return list(range(min(max(1, mn), n))) or [0]


def my_pol(o):
    try: return MY.agent(o)
    except Exception: return fv(o)


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


def opp_eval(state, p):
    me, op = state['players'][p], state['players'][1 - p]
    return 1000.0 * (len(op.get('prize') or []) - len(me.get('prize') or [])) + (_hp(me) - _hp(op))


def opp_greedy(obs_dict, opp_deck):
    """Deck-agnostic forward-sim greedy opponent."""
    sel = obs_dict.get('select')
    if sel is None: return None
    n = len(sel.get('option') or []); mn = int(sel.get('minCount', 1) or 1)
    if n < 2 or mn > 1:
        return fv(obs_dict)
    try:
        ob = api.to_observation_class(obs_dict); st = ob.current
        if st is None or getattr(ob, 'search_begin_input', None) is None:
            return fv(obs_dict)
        p = st.yourIndex; me, opp = st.players[p], st.players[1 - p]
        yd = list(opp_deck); yp = list(opp_deck)[:max(1, len(me.prize))]
        od = list(opp_deck); op_ = list(opp_deck)[:max(1, len(opp.prize))]
        oh = list(opp_deck)[:max(1, opp.handCount)]
        oa = [opp_deck[0]] if (len(opp.active) > 0 and opp.active[0] is None) else []
        root = api.search_begin(ob, yd, yp, od, op_, oh, oa)
        rid = root.searchId
        best_i, best_v = 0, -1e18
        for oi in range(n):
            try:
                child = raw_step(rid, [oi])
                if child.get('error', 0) != 0: continue
                ns = child['state']['observation'].get('current')
                v = opp_eval(ns, p) if ns else -1e17
            except Exception:
                v = -1e17
            if v > best_v: best_v, best_i = v, oi
        try: api.search_end()
        except Exception: pass
        return [best_i]
    except Exception:
        return fv(obs_dict)


def play(my_is_p0):
    if hasattr(MY, 'pre_turn'): MY.pre_turn = -1
    if my_is_p0:
        deck0, deck1 = ALAK, OPP_DECK
    else:
        deck0, deck1 = OPP_DECK, ALAK
    obs, sd = game.battle_start(deck0, deck1)
    if obs is None: return None
    for _ in range(2000):
        cur = obs.get('current')
        if cur is not None and cur.get('result', -1) != -1:
            return cur['result']
        if obs.get('select') is None: return None
        pp = cur['yourIndex'] if cur else 0
        my_turn = (pp == 0) == my_is_p0
        if my_turn:
            ch = my_pol(obs)
        else:
            ch = opp_greedy(obs, OPP_DECK)
        if ch is None: return None
        try: obs = game.battle_select(ch)
        except Exception: return None
    return None


def run(N):
    w = l = dr = er = 0; t0 = time.perf_counter()
    for g in range(N):
        my_is_p0 = (g % 2 == 0)
        r = play(my_is_p0)
        game.battle_finish()
        if r is None: er += 1; continue
        my_idx = 0 if my_is_p0 else 1
        if r == 2: dr += 1
        elif r == my_idx: w += 1
        else: l += 1
    tot = w + l + dr; wr = w / tot if tot else 0
    std = (wr * (1 - wr) / tot) ** 0.5 if tot else 0
    print(f'[{MY_DIR.split("/")[-1]} vs {OPP_DECK_JSON.split("/")[-1]}] my_W={w} L={l} draw={dr} err={er} '
          f'| my_winrate={wr:.3f} (+-{1.96*std:.3f}) | {time.perf_counter()-t0:.1f}s')


if __name__ == '__main__':
    run(N)
