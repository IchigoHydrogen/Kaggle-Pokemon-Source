"""v18d3: learn a LUCARIO-SPECIFIC value from self-play; candidate rollout eval.
Generate Lucario self-play (generic_fwdsim), extract Lucario-relevant features
per p0 decision + game outcome, train GBM value, report AUC (predictive?)."""
import sys, json, numpy as np, lightgbm as lgb
sys.path.insert(0,'/kaggle/working')
import generic_fwdsim_test as G
from sklearn.metrics import roc_auc_score
game=G.game
DECK=json.load(open('/kaggle/working/league_decks.json'))['Mega_Lucario']
MEGA=678; RIOLU=677; FIGHT=6
def fv(o):
    s=o.get('select')
    if s is None: return None
    mn=int(s.get('minCount',1) or 1); n=len(s.get('option',[]) or [])
    return list(range(min(max(1,mn),n))) or [0]
def feats(cur,p):
    me=cur['players'][p]; op=cur['players'][1-p]
    poks=([me['active'][0]] if me.get('active') and me['active'][0] else [])+[b for b in (me.get('bench') or []) if b]
    mega=[pk for pk in poks if pk.get('id')==MEGA]
    mega_fe=max([sum(1 for e in (pk.get('energies') or []) if e==FIGHT) for pk in mega] or [0])
    opa=(op.get('active') or [None]); ophp=int(opa[0].get('hp',0)) if (opa and opa[0]) else 0
    ma=(me.get('active') or [None]); mahp=int(ma[0].get('hp',0)) if (ma and ma[0]) else 0
    return [cur.get('turn',0), len(mega), mega_fe, sum(1 for pk in poks if pk.get('id')==RIOLU),
            sum(len(pk.get('energies') or []) for pk in poks), mahp, ophp,
            len([b for b in (me.get('bench') or []) if b]), len([b for b in (op.get('bench') or []) if b]),
            me.get('handCount',0), me.get('deckCount',0), len(me.get('prize') or []), len(op.get('prize') or [])]
FN=['turn','mega_count','mega_fight_energy','riolu_count','my_total_energy','my_active_hp','op_active_hp','my_bench','op_bench','my_hand','my_deck','my_prizes','op_prizes']
X=[]; meta=[]  # meta=(game,p)
N=int(sys.argv[1]) if len(sys.argv)>1 else 150
outcomes={}
for gi in range(N):
    obs,sd=game.battle_start(DECK,DECK); rows=[]
    for _ in range(2500):
        cur=obs.get('current')
        if cur is not None and cur.get('result',-1)!=-1:
            outcomes[gi]=cur['result']; break
        if obs.get('select') is None: break
        p=cur.get('yourIndex',0)
        if cur and cur.get('select') is None: pass
        sel=obs.get('select')
        if cur and sel and sel.get('context')==0:
            X.append(feats(cur,p)); meta.append((gi,p))
        ch=G.generic_fwdsim(obs,DECK) or fv(obs)
        try: obs=game.battle_select(ch)
        except Exception: break
    game.battle_finish()
y=np.array([1 if outcomes.get(g,-1)==p else 0 for (g,p) in meta])
X=np.array(X,dtype=float)
print('samples:',len(X),'win-rate:',round(y.mean(),3),'games:',len(outcomes))
# episode split
games=sorted(outcomes); vg=set(games[:len(games)//5])
val=np.array([g in vg for (g,p) in meta])
dtr=lgb.Dataset(X[~val],label=y[~val])
m=lgb.train({'objective':'binary','metric':'auc','verbose':-1,'num_leaves':31,'learning_rate':0.05,'min_data_in_leaf':30},dtr,num_boost_round=200)
auc=roc_auc_score(y[val],m.predict(X[val]))
print('Lucario value AUC=%.3f (val n=%d)'%(auc,val.sum()))
import pandas as pd
print('feat importance:', dict(zip(FN, m.feature_importance('gain').astype(int))))
m.save_model('/kaggle/working/lucario_value.txt')
print('DONE_V18D3')
