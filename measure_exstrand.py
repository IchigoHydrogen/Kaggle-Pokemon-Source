import sys, importlib.util
sys.path.insert(0,'/kaggle/input/competitions/pokemon-tcg-ai-battle/sample_submission')
import cg.game as game
from cg.api import all_card_data, all_attack
CT={c.cardId:c for c in all_card_data()}
ATK={a.attackId:(int(getattr(a,'damage',0) or 0), len(getattr(a,'energies',None) or [])) for a in all_attack()}
def min_atk_cost(cid):
    c=CT.get(cid)
    if not c: return 99
    costs=[ATK.get(a,(0,99))[1] for a in (getattr(c,'attacks',None) or [])]
    return min(costs) if costs else 99
AG=sys.argv[1]; N=int(sys.argv[2]) if len(sys.argv)>2 else 60
spec=importlib.util.spec_from_file_location('a', AG+'/main.py'); A=importlib.util.module_from_spec(spec); sys.modules['a']=A; spec.loader.exec_module(A)
deck=A.my_deck
def fv(o):
    s=o.get('select')
    if s is None: return None
    mn=int(s.get('minCount',1) or 1); n=len(s.get('option',[]) or [])
    return list(range(min(max(1,mn),n))) or [0]
strand_turns=0; total_myturns=0
for g in range(N):
    A.pre_turn=-1
    obs,sd=game.battle_start(deck,deck)
    if obs is None: continue
    seen=set()
    for _ in range(3000):
        cur=obs.get('current')
        if cur is not None and cur.get('result',-1)!=-1: break
        if obs.get('select') is None: break
        if cur and cur.get('select') is None: pass
        if cur:
            p=cur.get('yourIndex',0); me=cur['players'][p]
            t=cur.get('turn')
            act=me['active'][0] if me['active'] else None
            key=(g,t,p)
            if act and key not in seen:
                seen.add(key); total_myturns+=1
                aid=act.get('id'); ae=len(act.get('energies') or [])
                isex=getattr(CT.get(aid),'ex',False)
                bench_can=any(len(b.get('energies') or [])>=min_atk_cost(b.get('id')) for b in (me.get('bench') or []) if b)
                if isex and ae<min_atk_cost(aid) and bench_can:
                    strand_turns+=1
        ch=A.agent(obs) or fv(obs)
        try: obs=game.battle_select(ch)
        except Exception: break
    game.battle_finish()
print('agent=%s games=%d' % (AG,N))
print('  ex-stranded turns (active=ex can\'t attack, attacker on bench): %d / %d turns = %.1f%%' % (strand_turns,total_myturns,100*strand_turns/max(1,total_myturns)))
