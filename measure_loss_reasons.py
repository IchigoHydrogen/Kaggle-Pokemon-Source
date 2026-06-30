"""v11d measure-first: infer loss-reason distribution from real episode final states.

Loss reasons (engine): 1=opponent took all prizes (prize race), 2=deckout,
3=no active Pokemon (board collapse), 4=card effect. Episodes don't record the
reason code, so infer from the FINAL STATE:
  winner_prize==0 -> prize_race ; elif loser_deck==0 -> deckout ;
  elif loser has no active -> board_collapse ; else -> other.
Tells us which loss scenarios are common -> where v11d penalties have leverage.
"""
import json, glob, sys
from collections import Counter

EP_DIRS = ['/kaggle/input/competitions/pokemon-tcg-ai-battle-episodes-2026-06-26',
           '/kaggle/input/competitions/pokemon-tcg-ai-battle-episodes-2026-06-24']
N = int(sys.argv[1]) if len(sys.argv) > 1 else 400

files = []
for d in EP_DIRS:
    files += sorted(glob.glob(d + '/*.json'))
files = files[:N]

def final_state(j):
    for st in reversed(j['steps']):
        for a in st:
            ob = (a or {}).get('observation') or {}
            cur = ob.get('current')
            if cur and cur.get('players'):
                return cur
    return None

def npoke(pl):
    n = 0
    for p in (pl.get('active') or []):
        if p: n += 1
    for p in (pl.get('bench') or []):
        if p: n += 1
    return n

reasons = Counter(); ok = 0; turns = []
for f in files:
    try:
        j = json.load(open(f))
    except Exception:
        continue
    rw = j.get('rewards') or []
    if len(rw) != 2 or rw[0] == rw[1]:
        continue
    winner = 0 if rw[0] > rw[1] else 1
    loser = 1 - winner
    cur = final_state(j)
    if cur is None:
        continue
    pls = cur.get('players') or []
    if len(pls) < 2:
        continue
    w, l = pls[winner], pls[loser]
    wp = len(w.get('prize') or []); ld = l.get('deckCount', 0); la = npoke(l)
    if wp == 0:
        reasons['1_prize_race'] += 1
    elif ld == 0:
        reasons['2_deckout'] += 1
    elif la == 0 or len((l.get('active') or [])) == 0 or (l.get('active') or [None])[0] is None:
        reasons['3_board_collapse'] += 1
    else:
        reasons['4_other'] += 1
    turns.append(cur.get('turn', 0)); ok += 1

print(f'episodes classified: {ok}')
tot = max(1, ok)
for k in ['1_prize_race', '2_deckout', '3_board_collapse', '4_other']:
    print(f'  {k}: {reasons.get(k,0)} ({100*reasons.get(k,0)/tot:.1f}%)')
if turns:
    turns.sort(); print(f'game length (turns): median={turns[len(turns)//2]}, p90={turns[9*len(turns)//10]}')
