import pandas as pd, numpy as np, lightgbm as lgb
BASE='/kaggle/working/pokemon-20260628-v0-base28/pokemon-20260628-v0-base28-'
dec=pd.read_parquet(BASE+'decision_rows.parquet', columns=['decision_id','episode_id','player_archetype','context_name','num_options','min_count'])
# Lucario, real decisions (multi-option single-pick, like the agent's ctx0/promote)
luc=dec[(dec['player_archetype']=='Mega_Lucario') & (dec['num_options']>=2) & (dec['min_count']<=1)]
luc_ids=set(luc['decision_id'])
print('Lucario decisions (multi-opt single-pick):', len(luc_ids))
opt=pd.read_parquet(BASE+'option_rows.parquet')
opt=opt[opt['decision_id'].isin(luc_ids)].copy()
print('Lucario option rows:', len(opt))
# join episode for split
eps=opt['episode_id'].dropna().unique(); rng=np.random.RandomState(1); rng.shuffle(eps)
val_eps=set(eps[:len(eps)//5]); opt['val']=opt['episode_id'].isin(val_eps)
CAT=['option_type','card_id','target_card_id','attack_id','area','in_play_area']
NUM=['option_index','index','number_value','in_play_index','remain_damage_counter','remain_energy_cost']
for c in CAT: opt[c]=opt[c].astype('category')
for c in NUM: opt[c]=pd.to_numeric(opt[c],errors='coerce').fillna(-1.0)
opt=opt.sort_values('decision_id')
tr=opt[~opt['val']]; va=opt[opt['val']]
def grp(d): return d.groupby('decision_id',sort=False).size().values
dtr=lgb.Dataset(tr[CAT+NUM],label=tr['is_chosen'],group=grp(tr),categorical_feature=CAT)
m=lgb.train({'objective':'lambdarank','metric':'ndcg','ndcg_eval_at':[1],'lambdarank_truncation_level':1,'num_leaves':31,'learning_rate':0.05,'verbose':-1,'min_data_in_leaf':50},dtr,num_boost_round=250)
va=va.copy(); va['score']=m.predict(va[CAT+NUM])
# top1: among each decision's options, is the chosen one ranked #1?
def top1(g):
    return int(g.loc[g['score'].idxmax(),'is_chosen']==1)
acc=va.groupby('decision_id').apply(top1).mean()
# baseline: random top1 = mean(1/num_options)
nopt=va.groupby('decision_id').size(); base=(1/nopt).mean()
print('Lucario LGBM top1 accuracy=%.3f (val decisions=%d), random baseline=%.3f'%(acc, va['decision_id'].nunique(), base))
m.save_model('/kaggle/working/lucario_lgbm.txt')
print('saved lucario_lgbm.txt')
