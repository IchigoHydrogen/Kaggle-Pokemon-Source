import sys, importlib.util
sys.path.insert(0,'/kaggle/input/competitions/pokemon-tcg-ai-battle/sample_submission')
import cg.game as game
AG=sys.argv[1]; N=int(sys.argv[2]) if len(sys.argv)>2 else 60
spec=importlib.util.spec_from_file_location('a', AG+'/main.py')
A=importlib.util.module_from_spec(spec); sys.modules['a']=A; spec.loader.exec_module(A)
deck=A.my_deck
def fv(o):
    s=o.get('select')
    if s is None: return None
    mn=int(s.get('minCount',1) or 1); n=len(s.get('option',[]) or [])
    return list(range(min(max(1,mn),n))) or [0]
peakhands=[]; mindecks=[]; deckouts=0; played=0
for g in range(N):
    A.pre_turn=-1
    obs,sd=game.battle_start(deck,deck)
    if obs is None: continue
    peak=0; mind=99; last=None
    for _ in range(3000):
        cur=obs.get('current')
        if cur is not None and cur.get('result',-1)!=-1:
            last=cur; break
        if obs.get('select') is None: break
        cur=obs.get('current')
        if cur:
            p=cur.get('yourIndex',0); me=cur['players'][p]
            peak=max(peak, me.get('handCount',0)); mind=min(mind, me.get('deckCount',99))
        ch=A.agent(obs) or fv(obs)
        try: obs=game.battle_select(ch)
        except Exception: break
    game.battle_finish()
    if last is not None:
        played+=1; peakhands.append(peak); mindecks.append(mind)
        # deckout heuristic: someone's deck hit 0 during the game
        if mind==0: deckouts+=1
print('agent=%s  games=%d' % (AG, played))
print('  avg PEAK handCount (player to move): %.1f' % (sum(peakhands)/max(1,len(peakhands))))
print('  avg MIN deckCount reached         : %.1f' % (sum(mindecks)/max(1,len(mindecks))))
print('  %% games deck hit 0 (deckout risk) : %.1f%%' % (100*deckouts/max(1,played)))
print('  max hand observed                 : %d' % (max(peakhands) if peakhands else 0))
