"""Measure MY agent's own loss modes in self-play (does the Alakazam agent itself
lose by board-collapse / end benchless? -> tells us bench-insurance leverage)."""
import sys, importlib.util
from collections import Counter
sys.path.insert(0, '/kaggle/input/competitions/pokemon-tcg-ai-battle/sample_submission')
sys.path.insert(0, '/tmp/agent_v09d4')
import cg.game as game
spec = importlib.util.spec_from_file_location('v4', '/tmp/agent_v09d4/main.py')
A = importlib.util.module_from_spec(spec); sys.modules['v4'] = A; spec.loader.exec_module(A)
deck = A.my_deck

def fv(o):
    sel = o.get('select')
    if sel is None: return None
    mn = int(sel.get('minCount', 1) or 1); n = len(sel.get('option', []) or [])
    return list(range(min(max(1, mn), n))) or [0]

def npoke(pl):
    n = 0
    for p in (pl.get('active') or []):
        if p: n += 1
    for p in (pl.get('bench') or []):
        if p: n += 1
    return n

def play():
    A.pre_turn = -1
    obs, sd = game.battle_start(deck, deck)
    if obs is None: return None, None
    last = None
    for _ in range(2000):
        cur = obs.get('current')
        if cur is not None:
            last = cur
            if cur.get('result', -1) != -1:
                return cur['result'], cur
        if obs.get('select') is None: return None, last
        pp = cur['yourIndex'] if cur else 0
        ch = A.agent(obs) or fv(obs)
        try: obs = game.battle_select(ch)
        except Exception: return None, last
    return None, last

R = Counter(); benchless_at_loss = 0; ok = 0
N = int(sys.argv[1]) if len(sys.argv) > 1 else 100
for g in range(N):
    res, cur = play()
    game.battle_finish()
    if res is None or res == 2 or cur is None: continue
    w = res; l = 1 - w
    pls = cur.get('players') or []
    if len(pls) < 2: continue
    W, L = pls[w], pls[l]
    wp = len(W.get('prize') or []); ld = L.get('deckCount', 0); la = npoke(L)
    lbench = len([b for b in (L.get('bench') or []) if b])
    if lbench == 0: benchless_at_loss += 1
    if wp == 0: R['1_prize'] += 1
    elif ld == 0: R['2_deckout'] += 1
    elif la == 0 or not (L.get('active') or [None])[0]: R['3_collapse'] += 1
    else: R['4_other'] += 1
    ok += 1

print(f'self-play games (decisive): {ok}')
for k in ['1_prize', '2_deckout', '3_collapse', '4_other']:
    print(f'  {k}: {R.get(k,0)} ({100*R.get(k,0)/max(1,ok):.1f}%)')
print(f'loser was benchless at end: {benchless_at_loss} ({100*benchless_at_loss/max(1,ok):.1f}%)')
