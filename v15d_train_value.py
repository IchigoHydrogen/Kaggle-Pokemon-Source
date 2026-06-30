import pandas as pd, numpy as np, lightgbm as lgb, json
from sklearn.metrics import roc_auc_score
P='/kaggle/working/pokemon-20260628-v0-base28/pokemon-20260628-v0-base28-alakazam_value_dataset.parquet'
df=pd.read_parquet(P)
df['episode']=df['decision_id'].str.split(':').str[0]
eps=df['episode'].unique(); rng=np.random.RandomState(42); rng.shuffle(eps)
val_eps=set(eps[:len(eps)//5]); va=df['episode'].isin(val_eps); tr=~va
# SERVE-COMPUTABLE features only (computable from forward-sim sim state)
NUM=['turn','my_active_hp','my_active_energy_count','op_active_hp','op_active_energy_count',
     'my_bench_count','op_bench_count','my_alakazam_count','my_kadabra_count','my_abra_count','my_dudunsparce_count',
     'my_hand_count','op_hand_count','my_deck_count','op_deck_count','my_prizes_left','op_prizes_left','powerful_hand_damage_est']
CAT=['my_active_id','op_active_id','opponent_archetype']
for c in CAT: df[c]=df[c].astype('category')
feats=NUM+CAT
dtr=lgb.Dataset(df[feats][tr],label=df['won'][tr],categorical_feature=CAT)
dva=lgb.Dataset(df[feats][va],label=df['won'][va],categorical_feature=CAT,reference=dtr)
m=lgb.train({'objective':'binary','metric':'auc','verbose':-1,'num_leaves':31,'learning_rate':0.05,'feature_fraction':0.8,'min_data_in_leaf':50},
            dtr,num_boost_round=400,valid_sets=[dva],callbacks=[lgb.early_stopping(30,verbose=False)])
p=m.predict(df[feats][va]); print('serve-feature value AUC=%.4f (best_iter=%d)'%(roc_auc_score(df['won'][va],p),m.best_iteration))
early=va&(df['my_prizes_left']>=4)&(df['op_prizes_left']>=4)
print('  early-game AUC=%.4f'%roc_auc_score(df['won'][early],m.predict(df[feats][early])))
# category code maps (for serve-time encoding)
catmap={c:{str(k):int(v) for v,k in enumerate(df[c].cat.categories)} for c in CAT}
m.save_model('/kaggle/working/v15d_value.txt',num_iteration=m.best_iteration)
json.dump({'num':NUM,'cat':CAT,'catmap':catmap},open('/kaggle/working/v15d_value_meta.json','w'))
print('saved /kaggle/working/v15d_value.txt + meta. opponent_archetype cats:', list(df['opponent_archetype'].cat.categories))
