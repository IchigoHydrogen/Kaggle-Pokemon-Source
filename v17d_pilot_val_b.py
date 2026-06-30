import sys, json, time
sys.path.insert(0,'/kaggle/working')
import generic_fwdsim_test as G1
import generic_fwdsim_v2b as G2
game=G1.game
LEAGUE=json.load(open('/kaggle/working/league_decks.json'))
DECKS={k:LEAGUE[k] for k in ['Mega_Lucario','Hop_Trevenant','Archaludon_or_Other']}
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
N=int(sys.argv[1]) if len(sys.argv)>1 else 24
print('v2(terminal) vs v1(generic) same-deck pilot test, N=%d/deck'%N)
for dname,deck in DECKS.items():
    w=g=0
    for i in range(N):
        a_is_p0=(i%2==0)
        r=play((lambda o,dd=deck:G2.generic_fwdsim(o,dd)),(lambda o,dd=deck:G1.generic_fwdsim(o,dd)),deck,a_is_p0)
        game.battle_finish()
        if r is None: continue
        a_idx=0 if a_is_p0 else 1; g+=1
        if r==a_idx: w+=1
    print('  %-20s v2_winrate=%.3f (n=%d)'%(dname,w/max(1,g),g))
print('DONE_V2B')
