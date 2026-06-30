import sys, importlib.util
sys.path.insert(0,'/kaggle/working')
import generic_fwdsim_test as G
import v62_fwdsim as FS
sys.path.insert(0,'/kaggle/working/agent_v62lucario')
sp=importlib.util.spec_from_file_location('v62b','/kaggle/working/agent_v62lucario/main.py')
V=importlib.util.module_from_spec(sp); sp.loader.exec_module(V)
game=G.game; DECK=list(V.my_deck)
def fv(o):
    s=o.get('select')
    if s is None: return None
    mn=int(s.get('minCount',1) or 1); n=len(s.get('option',[]) or [])
    return list(range(min(max(1,mn),n))) or [0]
def rV(m):
    m.pre_turn=-1; m.ability_used=False; m.plan=m.AttackPlan()
def play(A,B,a0):
    rV(V); FS.V.pre_turn=-1; FS.V.ability_used=False; FS.V.plan=FS.V.AttackPlan()
    try: obs,sd=game.battle_start(DECK,DECK)
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
N=int(sys.argv[1]) if len(sys.argv)>1 else 30
w=g=0
for i in range(N):
    a=(i%2==0)
    r=play(FS.agent, V.agent, a)   # v62_fwdsim vs v62-alone
    game.battle_finish()
    if r is None: continue
    g+=1
    if r==(0 if a else 1): w+=1
print('v62_fwdsim vs v62-alone (does forward-sim help v62?): wr=%.3f (n=%d)'%(w/max(1,g),g))
print('DONE_V62FS')
