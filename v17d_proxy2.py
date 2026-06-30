# League field-proxy re-test with STRONG sensible_agent pilots (vs the weak generic_fwdsim v1)
import sys, json, importlib.util
sys.path.insert(0,'/kaggle/working')
sys.path.insert(0,'/kaggle/input/competitions/pokemon-tcg-ai-battle/sample_submission')
import cg.game as game
import sensible_pilot as S
L=json.load(open('/kaggle/working/league_decks.json'))
OPP={k:L[k] for k in ['Mega_Lucario','Hop_Trevenant','Archaludon_or_Other']}
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
print('LEAGUE re-test with STRONG sensible pilots. Kaggle: champion1023>exguard941>MC10 937>deckout896>v13d4 720')
res={}
for an,ad in AGENTS:
    m=load(ad,an.split('(')[0]); Adeck=list(m.my_deck); aw=tot=0; per={}
    for dn,dk in OPP.items():
        w=g=0
        for i in range(N):
            reset(m); a=(i%2==0)
            r=play(m.agent,Adeck,S.sensible_agent,dk,a)
            game.battle_finish()
            if r is None: continue
            g+=1
            if r==(0 if a else 1): w+=1
        per[dn]=round(w/max(1,g),2); aw+=w; tot+=g
    res[an]=aw/max(1,tot); print('  %-16s league_wr=%.3f | %s'%(an,res[an],per))
print('--- league ranking (sensible pilots) ---')
for a,wr in sorted(res.items(),key=lambda x:-x[1]): print('  %.3f %s'%(wr,a))
print('DONE_PROXY2')
