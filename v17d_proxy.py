import sys, time, json, importlib.util
sys.path.insert(0,'/kaggle/input/competitions/pokemon-tcg-ai-battle/sample_submission')
import cg.game as game
sys.path.insert(0,'/kaggle/working')
import generic_fwdsim_test as G
LEAGUE=json.load(open('/kaggle/working/league_decks.json'))
OPP={k:LEAGUE[k] for k in ['Mega_Lucario','Hop_Trevenant','Archaludon_or_Other']}
AGENTS=[('champion(1023)','/tmp/agent_v11d1_28thfull'),('exguard(941)','/tmp/agent_v11d4'),
        ('MC10(937)','/tmp/agent_v16d10'),('deckout(896)','/tmp/agent_v14d1'),('v13d4(720)','/tmp/agent_v13d4')]
def load(d,name):
    sys.path.insert(0,d); sp=importlib.util.spec_from_file_location(name,d+'/main.py')
    m=importlib.util.module_from_spec(sp); sys.modules[name]=m; sp.loader.exec_module(m); sys.path.pop(0); return m
def fv(o):
    s=o.get('select')
    if s is None: return None
    mn=int(s.get('minCount',1) or 1); n=len(s.get('option',[]) or [])
    return list(range(min(max(1,mn),n))) or [0]
def reset(m):
    for a in ('pre_turn','ability_used_dudunsparce','ability_used_fezandipiti'):
        if hasattr(m,a): setattr(m,a,-1 if a=='pre_turn' else False)
def play(Apol,Adeck,Bpol,Bdeck,A_is_p0):
    d0,d1=(Adeck,Bdeck) if A_is_p0 else (Bdeck,Adeck)
    try: obs,sd=game.battle_start(d0,d1)
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
N=int(sys.argv[1]) if len(sys.argv)>1 else 20
print('Kaggle order: champion1023 > exguard941 > MC10 937 > deckout896 > v13d4 720')
print('league N=%d/deck/agent'%N)
results={}
for aname,adir in AGENTS:
    m=load(adir,aname.split('(')[0])
    Adeck=list(m.my_deck)
    aw=tot=0; per={}
    for dname,ddeck in OPP.items():
        w=g=0
        for i in range(N):
            reset(m)
            if hasattr(G,'V4'): reset(G.V4)
            a_is_p0=(i%2==0)
            r=play(m.agent,Adeck,(lambda o,dd=ddeck:G.generic_fwdsim(o,dd)),ddeck,a_is_p0)
            game.battle_finish()
            if r is None: continue
            a_idx=0 if a_is_p0 else 1; g+=1
            if r==a_idx: w+=1
        per[dname]=w/max(1,g); aw+=w; tot+=g
    wr=aw/max(1,tot); results[aname]=wr
    print('  %-16s league_wr=%.3f  | %s' % (aname, wr, {k:round(v,2) for k,v in per.items()}))
print('--- league ranking (high=better) ---')
for a,wr in sorted(results.items(),key=lambda x:-x[1]): print('  %.3f %s'%(wr,a))
