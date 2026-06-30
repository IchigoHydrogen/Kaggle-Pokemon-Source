import sys, json
sys.path.insert(0,'/kaggle/working')
import generic_fwdsim_test as G
import sensible_pilot as S
game=G.game
L=json.load(open('/kaggle/working/league_decks.json'))
DECKS={k:L[k] for k in ['Mega_Lucario','Hop_Trevenant','Archaludon_or_Other']}
def fv(o):
    s=o.get('select')
    if s is None: return None
    mn=int(s.get('minCount',1) or 1); n=len(s.get('option',[]) or [])
    return list(range(min(max(1,mn),n))) or [0]
def play(Apol,Bpol,deck,A_is_p0):
    try: obs,sd=game.battle_start(deck,deck)
    except Exception: return None
    if obs is None: return None
    for _ in range(2500):
        cur=obs.get('current')
        if cur is not None and cur.get('result',-1)!=-1: return cur['result']
        if obs.get('select') is None: return None
        pp=cur['yourIndex'] if cur else 0
        a_turn=((pp==0)==A_is_p0)
        try: ch=(Apol(obs) if a_turn else Bpol(obs)) or fv(obs)
        except Exception: ch=fv(obs)
        if ch is None: return None
        try: obs=game.battle_select(ch)
        except Exception: return None
    return None
N=int(sys.argv[1]) if len(sys.argv)>1 else 40
print('sensible_agent vs generic_fwdsim, same deck (deck-agnostic strength check):')
for dn,dk in DECKS.items():
    w=g=0
    for i in range(N):
        a=(i%2==0)
        r=play(S.sensible_agent,(lambda o,d=dk:G.generic_fwdsim(o,d)),dk,a)
        game.battle_finish()
        if r is None: continue
        g+=1
        if r==(0 if a else 1): w+=1
    print('  %-20s sensible_wr=%.3f (n=%d)'%(dn,w/max(1,g),g))
print('DONE_SENS3')
