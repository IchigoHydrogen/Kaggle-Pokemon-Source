import sys, importlib.util
sys.path.insert(0,'/kaggle/working')
sys.path.insert(0,'/kaggle/input/competitions/pokemon-tcg-ai-battle/sample_submission')
import cg.game as game
sys.path.insert(0,'/kaggle/working/agent_v62lucario')
spv=importlib.util.spec_from_file_location('v62','/kaggle/working/agent_v62lucario/main.py')
V=importlib.util.module_from_spec(spv); spv.loader.exec_module(V)
def load(d,n):
    sys.path.insert(0,d); sp=importlib.util.spec_from_file_location(n,d+'/main.py')
    m=importlib.util.module_from_spec(sp); sys.modules[n]=m; sp.loader.exec_module(m); sys.path.pop(0); return m
CH=load('/tmp/agent_v11d1_28thfull','champ')
V_DECK=list(V.my_deck); CH_DECK=list(CH.my_deck)
def fv(o):
    s=o.get('select')
    if s is None: return None
    mn=int(s.get('minCount',1) or 1); n=len(s.get('option',[]) or [])
    return list(range(min(max(1,mn),n))) or [0]
def rV(): V.pre_turn=-1; V.ability_used=False; V.plan=V.AttackPlan()
def rC():
    for a,v in [('pre_turn',-1),('ability_used_dudunsparce',False),('ability_used_fezandipiti',False)]:
        if hasattr(CH,a): setattr(CH,a,v)
def play(a0):  # a0: True => v62 is p0
    rV(); rC()
    d0,d1=(V_DECK,CH_DECK) if a0 else (CH_DECK,V_DECK)
    try: obs,sd=game.battle_start(d0,d1)
    except Exception: return None
    if obs is None: return None
    for _ in range(2500):
        cur=obs.get('current')
        if cur is not None and cur.get('result',-1)!=-1: return cur['result']
        if obs.get('select') is None: return None
        pp=cur['yourIndex'] if cur else 0
        vturn=((pp==0)==a0)
        try: ch=(V.agent(obs) if vturn else CH.agent(obs)) or fv(obs)
        except Exception: ch=fv(obs)
        if ch is None: return None
        try: obs=game.battle_select(ch)
        except Exception: return None
    return None
N=int(sys.argv[1]) if len(sys.argv)>1 else 50
w=g=0
for i in range(N):
    a=(i%2==0)
    r=play(a)
    game.battle_finish()
    if r is None: continue
    g+=1
    if r==(0 if a else 1): w+=1
print('v62-Lucario vs champion(Alakazam 1023): v62_wr=%.3f (n=%d) [non-predictive of Kaggle field, characterizes matchup]'%(w/max(1,g),g))
print('DONE_V62CHAMP')
