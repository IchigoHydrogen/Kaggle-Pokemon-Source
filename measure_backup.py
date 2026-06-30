import sys, importlib.util
sys.path.insert(0,'/kaggle/input/competitions/pokemon-tcg-ai-battle/sample_submission')
import cg.game as game
PSY={741,742,743}
AG=sys.argv[1]; N=int(sys.argv[2]) if len(sys.argv)>2 else 60
spec=importlib.util.spec_from_file_location('a', AG+'/main.py'); A=importlib.util.module_from_spec(spec); sys.modules['a']=A; spec.loader.exec_module(A)
deck=A.my_deck
def fv(o):
    s=o.get('select')
    if s is None: return None
    mn=int(s.get('minCount',1) or 1); n=len(s.get('option',[]) or [])
    return list(range(min(max(1,mn),n))) or [0]
turns=0; with_backup=0; total_bench_e=0; total_active_overE=0
for g in range(N):
    A.pre_turn=-1
    obs,sd=game.battle_start(deck,deck)
    if obs is None: continue
    seen=set()
    for _ in range(3000):
        cur=obs.get('current')
        if cur is not None and cur.get('result',-1)!=-1: break
        if obs.get('select') is None: break
        if cur:
            p=cur.get('yourIndex',0); me=cur['players'][p]; t=cur.get('turn')
            key=(g,t,p)
            if key not in seen:
                seen.add(key); turns+=1
                bench=[b for b in (me.get('bench') or []) if b]
                if any(b.get('id') in PSY and len(b.get('energies') or [])>=1 for b in bench): with_backup+=1
                total_bench_e+=sum(len(b.get('energies') or []) for b in bench)
                act=me['active'][0] if me['active'] else None
                if act and act.get('id') in PSY: total_active_overE+=max(0,len(act.get('energies') or [])-1)
        ch=A.agent(obs) or fv(obs)
        try: obs=game.battle_select(ch)
        except Exception: break
    game.battle_finish()
print('agent=%s games=%d turns=%d' % (AG,N,turns))
print('  %% my-turns with READY bench backup (psy >=1E): %.1f%%' % (100*with_backup/max(1,turns)))
print('  avg bench energy per turn: %.2f' % (total_bench_e/max(1,turns)))
print('  total wasted active energy (psy active >1E): %d (=%.2f/turn)' % (total_active_overE, total_active_overE/max(1,turns)))
