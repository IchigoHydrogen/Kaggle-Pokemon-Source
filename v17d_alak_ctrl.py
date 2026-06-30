import pandas as pd, numpy as np, lightgbm as lgb
BASE='/kaggle/working/pokemon-20260628-v0-base28/pokemon-20260628-v0-base28-'
dec=pd.read_parquet(BASE+'decision_rows.parquet', columns=['decision_id','player_archetype','num_options','min_count'])
luc=dec[(dec['player_archetype']=='Alakazam')&(dec['num_options']>=2)&(dec['min_count']<=1)]
ids=set(luc['decision_id']); print('Lucario multi-opt decisions:', len(ids))
ss=pd.read_parquet(BASE+'state_summary.parquet')
print('state_summary cols:', list(ss.columns))
ss=ss[ss['decision_id'].isin(ids)]
opt=pd.read_parquet(BASE+'option_rows.parquet')
opt=opt[opt['decision_id'].isin(ids)].copy()
# join board-state features onto each option (broadcast per decision)
sscols=[c for c in ss.columns if c not in ('episode_id','player_index')]
opt=opt.merge(ss[sscols], on='decision_id', how='left', suffixes=('','_ss'))
print('joined Lucario option rows:', opt.shape)
eps=opt['episode_id'].dropna().unique(); rng=np.random.RandomState(1); rng.shuffle(eps)
val=set(eps[:len(eps)//5]); opt['val']=opt['episode_id'].isin(val)
OPT_CAT=['option_type','card_id','target_card_id','attack_id','area','in_play_area']
SS_CAT=[c for c in ['my_active_id','op_active_id','stadium_id','context_name'] if c in opt.columns]
CAT=[c for c in OPT_CAT+SS_CAT if c in opt.columns]
DROP=set(['decision_id','episode_id','is_chosen','val','player_index']+CAT)
NUM=[c for c in opt.columns if c not in DROP and opt[c].dtype!=object]
for c in CAT: opt[c]=opt[c].astype('category')
for c in NUM: opt[c]=pd.to_numeric(opt[c],errors='coerce').fillna(-1.0)
opt=opt.sort_values('decision_id')
tr=opt[~opt['val']]; va=opt[opt['val']]
g=lambda d:d.groupby('decision_id',sort=False).size().values
dtr=lgb.Dataset(tr[CAT+NUM],label=tr['is_chosen'],group=g(tr),categorical_feature=CAT)
m=lgb.train({'objective':'lambdarank','metric':'ndcg','ndcg_eval_at':[1],'lambdarank_truncation_level':1,'num_leaves':63,'learning_rate':0.05,'verbose':-1,'min_data_in_leaf':50},dtr,num_boost_round=300)
va=va.copy(); va['s']=m.predict(va[CAT+NUM])
acc=va.groupby('decision_id').apply(lambda x:int(x.loc[x['s'].idxmax(),'is_chosen']==1)).mean()
base=(1/va.groupby('decision_id').size()).mean()
print('CONTROL Alakazam LGBM top1=%.3f (vs generic-only 0.199, random %.3f), n=%d feats=%d'%(acc,base,va['decision_id'].nunique(),len(CAT+NUM)))
imp=pd.Series(m.feature_importance('gain'),index=CAT+NUM).sort_values(ascending=False)
print('top feats:', imp.head(10).to_dict())
