import sys, json
sys.path.insert(0,'/kaggle/working')
import generic_fwdsim_test as G
import lucario_value_fwdsim as LV
game=G.game
DECK=json.load(open('/kaggle/working/league_decks.json'))['Mega_Lucario']
def fv(o):
    s=o.get('select')
    if s is None: return None
    mn=int(s.get('minCount',1) or 1); n=len(s.get('option',[]) or [])
    return list(range(min(max(1,mn),n))) or [0]
def play(A,B,deck,a0):
    try: obs,sd=game.battle_start(deck,deck)
    except Exception: return None
    if obs is None: return None
    for _ in range(2500):
        cur=obs.get('current')
        if cur is not None and cur.get('result',-1)!=-1: return cur['result']
        if obs.get('select') is None: return None
        pp=cur['yourIndex'] if cur else 0
        a=((pp==0)==a0)
        try: ch=(A(obs) if a else B(obs)) or fv(obs)
        except Exception: ch=fv(obs)
        if ch is None: return None
        try: obs=game.battle_select(ch)
        except Exception: return None
    return None
N=int(sys.argv[1]) if len(sys.argv)>1 else 50
w=g=0
for i in range(N):
    a=(i%2==0)
    r=play((lambda o:LV.lucario_value_fwdsim(o,DECK)),(lambda o:G.generic_fwdsim(o,DECK)),DECK,a)
    game.battle_finish()
    if r is None: continue
    g+=1
    if r==(0 if a else 1): w+=1
print('lucario_VALUE_fwdsim vs generic_fwdsim, Lucario: wr=%.3f (n=%d)'%(w/max(1,g),g))
print('DONE_V18D3B')
