import pandas as pd, numpy as np, lightgbm as lgb
from sklearn.metrics import roc_auc_score
P='/kaggle/working/pokemon-20260628-v0-base28/pokemon-20260628-v0-base28-alakazam_value_dataset.parquet'
df=pd.read_parquet(P)
df['episode']=df['decision_id'].str.split(':').str[0]
# episode-level split (avoid leakage)
eps=df['episode'].unique()
rng=np.random.RandomState(42); rng.shuffle(eps)
val_eps=set(eps[:len(eps)//5])
df['is_val']=df['episode'].isin(val_eps)
PRIZE=['my_prizes_left','op_prizes_left']
CAT=['my_active_id','op_active_id','stadium_id','context_name','tier','matchup_label','opponent_archetype']
NUM=['turn','turn_action_count','my_active_hp','my_active_energy_count','op_active_hp','op_active_energy_count',
     'my_bench_count','op_bench_count','my_alakazam_count','my_kadabra_count','my_abra_count','my_dudunsparce_count',
     'my_hand_count','op_hand_count','my_deck_count','op_deck_count','powerful_hand_damage_est','deckout_risk',
     'powerful_hand_can_ko_active','deckout_risk_feature']
for c in CAT: df[c]=df[c].astype('category')
def train_eval(feats, name, tr, va):
    X=df[feats]; y=df['won']
    cats=[c for c in feats if c in CAT]
    dtr=lgb.Dataset(X[tr],label=y[tr],categorical_feature=cats)
    m=lgb.train({'objective':'binary','metric':'auc','verbose':-1,'num_leaves':31,'learning_rate':0.05,'feature_fraction':0.8},
                dtr,num_boost_round=300)
    p=m.predict(X[va])
    auc=roc_auc_score(y[va],p)
    print('  [%s] valid AUC=%.4f (n_val=%d)' % (name,auc,va.sum()))
    return m,p
tr=~df['is_val']; va=df['is_val']
print('=== FULL model (prizes + positional + cat) ===')
m_full,_=train_eval(NUM+PRIZE+CAT,'full',tr,va)
print('=== POSITIONAL-only (NO prizes) ===')
m_pos,_=train_eval(NUM+CAT,'positional-no-prizes',tr,va)
print('=== EARLY-GAME subset (both prizes_left>=4): can position foreshadow winner? ===')
early=df['is_val'] & (df['my_prizes_left']>=4) & (df['op_prizes_left']>=4)
# full model AUC on early subset
Xv=df[NUM+PRIZE+CAT]; pv=m_full.predict(Xv[early]); print('  [full on early] AUC=%.4f (n=%d)'%(roc_auc_score(df['won'][early],pv),early.sum()))
Xv2=df[NUM+CAT]; pv2=m_pos.predict(Xv2[early]); print('  [positional-only on early] AUC=%.4f'%roc_auc_score(df['won'][early],pv2))
print('=== top feature importances (full model, by gain) ===')
imp=pd.Series(m_full.feature_importance('gain'),index=NUM+PRIZE+CAT).sort_values(ascending=False)
print(imp.head(15).to_string())
