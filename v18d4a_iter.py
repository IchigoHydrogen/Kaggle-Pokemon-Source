"""v18d4a: AlphaZero-style value iteration. Generate self-play with the v1
VALUE-pilot (better play than greedy), retrain value v2, report AUC."""
import sys, json, numpy as np, lightgbm as lgb
sys.path.insert(0,'/kaggle/working')
import generic_fwdsim_test as G
import lucario_value_fwdsim as LV
from sklearn.metrics import roc_auc_score
game=G.game
DECK=json.load(open('/kaggle/working/league_decks.json'))['Mega_Lucario']
feats=LV.feats
def fv(o):
    s=o.get('select')
    if s is None: return None
    mn=int(s.get('minCount',1) or 1); n=len(s.get('option',[]) or [])
    return list(range(min(max(1,mn),n))) or [0]
X=[]; meta=[]; outcomes={}
N=int(sys.argv[1]) if len(sys.argv)>1 else 100
for gi in range(N):
    obs,sd=game.battle_start(DECK,DECK)
    for _ in range(2500):
        cur=obs.get('current')
        if cur is not None and cur.get('result',-1)!=-1:
            outcomes[gi]=cur['result']; break
        if obs.get('select') is None: break
        p=cur.get('yourIndex',0); sel=obs.get('select')
        if cur and sel and sel.get('context')==0:
            X.append(feats(cur,p)); meta.append((gi,p))
        ch=LV.lucario_value_fwdsim(obs,DECK) or fv(obs)   # v1 VALUE-pilot policy
        try: obs=game.battle_select(ch)
        except Exception: break
    game.battle_finish()
y=np.array([1 if outcomes.get(g,-1)==p else 0 for (g,p) in meta])
X=np.array(X,dtype=float)
print('v2 samples:',len(X),'win-rate:',round(y.mean(),3),'games:',len(outcomes))
games=sorted(outcomes); vg=set(games[:len(games)//5])
val=np.array([g in vg for (g,p) in meta])
dtr=lgb.Dataset(X[~val],label=y[~val])
m=lgb.train({'objective':'binary','metric':'auc','verbose':-1,'num_leaves':31,'learning_rate':0.05,'min_data_in_leaf':30},dtr,num_boost_round=200)
print('Lucario value v2 AUC=%.3f (val n=%d) [v1 was 0.753]'%(roc_auc_score(y[val],m.predict(X[val])),val.sum()))
m.save_model('/kaggle/working/lucario_value_v2.txt')
print('DONE_V18D4A')
