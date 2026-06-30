import sys, importlib.util, copy
from collections import Counter
sys.path.insert(0,'/kaggle/input/competitions/pokemon-tcg-ai-battle/sample_submission')
sys.path.insert(0,'/tmp/agent_v11d1_28thfull')
import cg.game as game
spec=importlib.util.spec_from_file_location('m','/tmp/agent_v11d1_28thfull/main.py')
m=importlib.util.module_from_spec(spec); sys.modules['m']=m; spec.loader.exec_module(m)
deck=m.my_deck
def fv(o):
    s=o.get('select');
    if s is None: return None
    mn=int(s.get('minCount',1) or 1); n=len(s.get('option',[]) or [])
    return list(range(min(max(1,mn),n))) or [0]
# capture a few ctx0 multi-option decisions from a self-play game
captured=[]
m.pre_turn=-1
obs,sd=game.battle_start(deck,deck)
for _ in range(2000):
    cur=obs.get('current')
    if cur and cur.get('result',-1)!=-1: break
    sel=obs.get('select')
    if sel is None: break
    if cur and sel.get('context')==0 and len(sel.get('option') or [])>=3 and int(sel.get('minCount',1) or 1)<=1 and cur.get('turn',0)>=3 and len(captured)<4:
        captured.append(copy.deepcopy(obs))
    ch=m.agent(obs) or fv(obs)
    try: obs=game.battle_select(ch)
    except Exception: break
game.battle_finish()
print('captured %d ctx0 decisions' % len(captured))
# for each, call agent K times -> action distribution (stochastic?)
for i,o in enumerate(captured):
    acts=[]
    for k in range(12):
        m.pre_turn=-1
        try: a=m.agent(copy.deepcopy(o))
        except Exception as e: a=('ERR',str(e)[:40])
        acts.append(tuple(a) if isinstance(a,list) else a)
    c=Counter(acts)
    nopt=len(o['select']['option']); turn=o['current'].get('turn')
    print('  obs%d turn=%d opts=%d -> distinct_actions=%d dist=%s' % (i,turn,nopt,len(c),dict(c)))
