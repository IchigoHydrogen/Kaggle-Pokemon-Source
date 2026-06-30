import sys, importlib.util
from collections import Counter
sys.path.insert(0, '/kaggle/input/competitions/pokemon-tcg-ai-battle/sample_submission')
sys.path.insert(0, '/tmp/agent_v09d4')
import cg.game as game
spec = importlib.util.spec_from_file_location('a4', '/tmp/agent_v09d4/main.py')
A = importlib.util.module_from_spec(spec); sys.modules['a4']=A; spec.loader.exec_module(A)
my_deck = A.my_deck
def fv(o):
    sel=o.get('select')
    if sel is None: return None
    mn=int(sel.get('minCount',1) or 1); n=len(sel.get('option',[]) or [])
    return list(range(min(max(1,mn),n))) or [0]
multi=Counter(); allc=Counter()
for g in range(12):
    A.pre_turn=-1
    obs,sd=game.battle_start(my_deck,my_deck)
    for _ in range(2000):
        cur=obs.get('current')
        if cur is not None and cur.get('result',-1)!=-1: break
        sel=obs.get('select')
        if sel is None: break
        ctx=sel.get('context'); n=len(sel.get('option',[]) or [])
        allc[ctx]+=1
        if n>=2: multi[ctx]+=1
        try: obs=game.battle_select(A.agent(obs) or fv(obs))
        except Exception: break
    game.battle_finish()
print('ALL decisions by context:', dict(sorted(allc.items(), key=lambda x:-x[1])))
print('MULTI-option (>=2) decisions by context:', dict(sorted(multi.items(), key=lambda x:-x[1])))
tot_multi=sum(multi.values()); 
print('ctx0 share of multi-option decisions:', round(multi.get(0,0)/max(1,tot_multi),3))
