import sys, json
from collections import Counter
sys.path.insert(0,'/kaggle/working')
import generic_fwdsim_test as G
import sensible_pilot as S
from cg.api import OptionType, SelectContext, to_observation_class
game=G.game
DECK=json.load(open('/kaggle/working/league_decks.json'))['Mega_Lucario']
def fv(o):
    s=o.get('select')
    if s is None: return None
    mn=int(s.get('minCount',1) or 1); n=len(s.get('option',[]) or [])
    return list(range(min(max(1,mn),n))) or [0]
# instrument: sensible_agent plays both sides; log chosen option type per MAIN decision + result
act_types=Counter(); ctx_seen=Counter()
def logged_sensible(obs):
    ch=S.sensible_agent(obs)
    try:
        ob=to_observation_class(obs); sel=ob.select
        ctx_seen[str(sel.context)]+=1
        if ch:
            o=sel.option[ch[0]]
            act_types[(str(sel.context),str(getattr(o,'type',None)))]+=1
    except Exception: pass
    return ch
prizes_taken=[]; turns=[]
for gi in range(6):
    obs,sd=game.battle_start(DECK,DECK); t=0
    for _ in range(2500):
        cur=obs.get('current')
        if cur is not None and cur.get('result',-1)!=-1:
            me=cur['players'][0]; prizes_taken.append(6-len(me.get('prize') or [])); turns.append(cur.get('turn')); break
        if obs.get('select') is None: break
        ch=logged_sensible(obs) or fv(obs)
        try: obs=game.battle_select(ch)
        except Exception: break
    game.battle_finish()
print('=== sensible_agent action distribution (chosen option type per context) ===')
for k,v in act_types.most_common(20): print('  %s : %d'%(k,v))
print('avg prizes taken (p0):', sum(prizes_taken)/max(1,len(prizes_taken)), 'avg final turn:', sum(turns)/max(1,len(turns)))
