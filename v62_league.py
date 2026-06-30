# League field-proxy test with STRONG v62 Lucario pilot (vs the weak generic_fwdsim v1)
import sys, importlib.util
sys.path.insert(0,'/kaggle/working')
sys.path.insert(0,'/kaggle/input/competitions/pokemon-tcg-ai-battle/sample_submission')
import cg.game as game
sys.path.insert(0,'/kaggle/working/agent_v62lucario')
spv=importlib.util.spec_from_file_location('v62','/kaggle/working/agent_v62lucario/main.py')
V=importlib.util.module_from_spec(spv); spv.loader.exec_module(V)
V_DECK=list(V.my_deck)
AGENTS=[('champion(1023)','/tmp/agent_v11d1_28thfull'),('exguard(941)','/tmp/agent_v11d4'),
        ('MC10(937)','/tmp/agent_v16d10'),('deckout(896)','/tmp/agent_v14d1'),('v13d4(720)','/tmp/agent_v13d4')]
def load(d,n):
    sys.path.insert(0,d); sp=importlib.util.spec_from_file_location(n,d+'/main.py')
    m=importlib.util.module_from_spec(sp); sys.modules[n]=m; sp.loader.exec_module(m); sys.path.pop(0); return m
def fv(o):
    s=o.get('select')
    if s is None: return None
    mn=int(s.get('minCount',1) or 1); n=len(s.get('option',[]) or [])
    return list(range(min(max(1,mn),n))) or [0]
def reset(m):
    for a,v in [('pre_turn',-1),('ability_used_dudunsparce',False),('ability_used_fezandipiti',False),('ability_used',False)]:
        if hasattr(m,a): setattr(m,a,v)
    if hasattr(m,'plan') and hasattr(m,'AttackPlan'): m.plan=m.AttackPlan()
def play(Apol,Adeck,Bpol,Bdeck,a0):
    d0,d1=(Adeck,Bdeck) if a0 else (Bdeck,Adeck)
    try: obs,sd=game.battle_start(d0,d1)
    except Exception: return None
    if obs is None: return None
    for _ in range(2500):
        cur=obs.get('current')
        if cur is not None and cur.get('result',-1)!=-1: return cur['result']
        if obs.get('select') is None: return None
        pp=cur['yourIndex'] if cur else 0
        a=((pp==0)==a0)
        try: ch=(Apol(obs) if a else Bpol(obs)) or fv(obs)
        except Exception: ch=fv(obs)
        if ch is None: return None
        try: obs=game.battle_select(ch)
        except Exception: return None
    return None
N=int(sys.argv[1]) if len(sys.argv)>1 else 30
print('LEAGUE test: 5 Alakazam agents vs STRONG v62-Lucario. Kaggle: champ1023>exguard941>MC10 937>deckout896>v13d4 720')
res={}
for an,ad in AGENTS:
    try: m=load(ad,an.split('(')[0])
    except Exception as e: print('  skip',an,e); continue
    Adeck=list(m.my_deck); w=g=0
    for i in range(N):
        a=(i%2==0); reset(m); reset(V)
        r=play(m.agent,Adeck,V.agent,V_DECK,a)
        game.battle_finish()
        if r is None: continue
        g+=1
        if r==(0 if a else 1): w+=1
    res[an]=w/max(1,g); print('  %-16s vs_v62_wr=%.3f (n=%d)'%(an,res[an],g))
print('--- ranking vs v62 (strong pilot) ---')
for a,wr in sorted(res.items(),key=lambda x:-x[1]): print('  %.3f %s'%(wr,a))
print('DONE_V62LEAGUE')
