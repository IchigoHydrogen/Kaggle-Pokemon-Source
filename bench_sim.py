import sys, time, os
sys.path.insert(0,'/kaggle/input/competitions/pokemon-tcg-ai-battle/sample_submission')
sys.path.insert(0,'/tmp/agent_v11d1_28thfull')
import importlib.util as u
# get a valid deck (light: just my_deck list) without forcing heavy agent logic at runtime
spec=u.spec_from_file_location('m','/tmp/agent_v11d1_28thfull/main.py'); m=u.module_from_spec(spec); sys.modules['m']=m; spec.loader.exec_module(m)
DECK=list(m.my_deck)
import cg.game as game
def fv(o):
    s=o.get('select')
    if s is None: return None
    mn=int(s.get('minCount',1) or 1); n=len(s.get('option',[]) or [])
    return list(range(min(max(1,mn),n))) or [0]
def worker(duration):
    steps=0; games=0; t_end=time.time()+duration
    while time.time()<t_end:
        try:
            obs,sd=game.battle_start(DECK,DECK)
        except Exception: break
        if obs is None: continue
        for _ in range(4000):
            cur=obs.get('current')
            if cur and cur.get('result',-1)!=-1: break
            if obs.get('select') is None: break
            ch=fv(obs)
            if ch is None: break
            try: obs=game.battle_select(ch)
            except Exception: break
            steps+=1
        try: game.battle_finish()
        except Exception: pass
        games+=1
    return steps,games
if __name__=='__main__':
    mode=sys.argv[1] if len(sys.argv)>1 else 'single'
    if mode=='single':
        D=12
        s,g=worker(D)
        print('SINGLE: %.0f steps/sec, %.1f games/sec, avg %.0f steps/game (%ds)' % (s/D, g/D, s/max(1,g), D))
    else:
        import concurrent.futures as cf
        D=8
        for K in [4,8,16,24,32]:
            t0=time.time()
            with cf.ProcessPoolExecutor(max_workers=K) as ex:
                res=list(ex.map(worker,[D]*K))
            el=time.time()-t0
            tot_s=sum(r[0] for r in res); tot_g=sum(r[1] for r in res)
            print('K=%2d: %8.0f steps/sec (aggregate), %6.1f games/sec, wall=%.1fs' % (K, tot_s/el, tot_g/el, el))
