import pandas as pd, numpy as np, lightgbm as lgb, json
from sklearn.metrics import roc_auc_score
P='/kaggle/working/pokemon-20260628-v0-base28/pokemon-20260628-v0-base28-alakazam_value_dataset.parquet'
df=pd.read_parquet(P)
df['episode']=df['decision_id'].str.split(':').str[0]
eps=df['episode'].unique(); rng=np.random.RandomState(42); rng.shuffle(eps)
val_eps=set(eps[:len(eps)//5]); va=df['episode'].isin(val_eps); tr=~va
NUM=['turn','my_active_hp','my_active_energy_count','op_active_hp','op_active_energy_count',
     'my_bench_count','op_bench_count','my_alakazam_count','my_kadabra_count','my_abra_count','my_dudunsparce_count',
     'my_hand_count','op_hand_count','my_deck_count','op_deck_count','my_prizes_left','op_prizes_left','powerful_hand_damage_est']
X=df[NUM].fillna(-1.0)
dtr=lgb.Dataset(X[tr],label=df['won'][tr]); dva=lgb.Dataset(X[va],label=df['won'][va],reference=dtr)
m=lgb.train({'objective':'binary','metric':'auc','verbose':-1,'num_leaves':31,'learning_rate':0.05,'feature_fraction':0.8,'min_data_in_leaf':50},
            dtr,num_boost_round=400,valid_sets=[dva],callbacks=[lgb.early_stopping(30,verbose=False)])
p=m.predict(X[va]); print('NUM-only AUC=%.4f (best_iter=%d)'%(roc_auc_score(df['won'][va],p),m.best_iteration))
early=va&(df['my_prizes_left']>=4)&(df['op_prizes_left']>=4)
print('  early-game AUC=%.4f'%roc_auc_score(df['won'][early],m.predict(X[early])))
m.save_model('/kaggle/working/v15d_value.txt',num_iteration=m.best_iteration)
json.dump({'num':NUM},open('/kaggle/working/v15d_value_meta.json','w'))
imp=pd.Series(m.feature_importance('gain'),index=NUM).sort_values(ascending=False)
print('top features:'); print(imp.head(10).to_string())
print('saved NUM-only model')
