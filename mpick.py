import sys, importlib.util
from collections import Counter
sys.path.insert(0,'/kaggle/input/competitions/pokemon-tcg-ai-battle/sample_submission')
sys.path.insert(0,'/tmp/agent_v11d1_28thfull')
import cg.game as game
spec=importlib.util.spec_from_file_location('a','/tmp/agent_v11d1_28thfull/main.py');A=importlib.util.module_from_spec(spec);sys.modules['a']=A;spec.loader.exec_module(A)
deck=A.my_deck
def fv(o):
    sel=o.get('select')
    if sel is None:return None
    mn=int(sel.get('minCount',1) or 1);n=len(sel.get('option',[]) or [])
    return list(range(min(max(1,mn),n))) or [0]
ctx0=Counter()
for g in range(15):
    A.pre_turn=-1
    obs,sd=game.battle_start(deck,deck)
    for _ in range(2000):
        cur=obs.get('current')
        if cur is not None and cur.get('result',-1)!=-1:break
        sel=obs.get('select')
        if sel is None:break
        if sel.get('context')==0:
            n=len(sel.get('option',[]) or []);mn=int(sel.get('minCount',1) or 1)
            if n>=2:
                ctx0['total']+=1
                if mn>1:ctx0['multipick']+=1
                else:ctx0['singlepick']+=1
        try:obs=game.battle_select(A.agent(obs) or fv(obs))
        except:break
    game.battle_finish()
print('ctx0 multi-option decisions:',ctx0['total'])
print('  single-pick (forward-sim optimizes):',ctx0['singlepick'],f"({100*ctx0['singlepick']/max(1,ctx0['total']):.1f}%)")
print('  multi-pick  (forward-sim SKIPS->base):',ctx0['multipick'],f"({100*ctx0['multipick']/max(1,ctx0['total']):.1f}%)")
