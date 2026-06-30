"""Gold-unlock step 1: can we learn a Hop_Trevenant imitation model from logs?

Join OPTION_ROWS (generic option features) with DECISION_ROWS (player_archetype),
filter to Hop_Trevenant UNKNOWN_0 decisions, train an LGBM LambdaRank imitation
model (predict is_chosen, group by decision), report holdout top1. If Hop imitation
top1 is decent (~0.5, like Alakazam), a Hop imitation agent is viable -> strong
diverse opponent -> validation unlock for opponent-aware play.
"""
import sys
import numpy as np
import pandas as pd
import lightgbm as lgb

BASE = '/kaggle/working/pokemon-20260627-v0-09d2/pokemon-20260627-v0-09d2'
ARCH = sys.argv[1] if len(sys.argv) > 1 else 'Hop_Trevenant'

opt = pd.read_parquet(BASE + '-option_rows.parquet')
dec = pd.read_parquet(BASE + '-decision_rows.parquet')[
    ['decision_id', 'player_archetype', 'opponent_archetype', 'tier', 'won', 'num_options']]
df = opt.merge(dec, on='decision_id', how='inner')
df = df[(df['player_archetype'] == ARCH) & (df['context_name'] == 'UNKNOWN_0')].copy()
print(f'{ARCH} UNKNOWN_0 option rows: {len(df)}, decisions: {df["decision_id"].nunique()}')

# generic features
NUM = ['option_index', 'num_options', 'card_id', 'target_card_id', 'attack_id',
       'number_value', 'in_play_index', 'remain_damage_counter', 'remain_energy_cost',
       'player_index_option']
CAT = ['option_type', 'area', 'in_play_area']
for c in NUM:
    if c not in df.columns: df[c] = 0.0
    df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0.0)
cat_maps = {}
for c in CAT:
    if c not in df.columns: df[c] = 'NA'
    df[c] = df[c].astype(str).fillna('NA')
    u = ['UNK'] + sorted(df[c].unique().tolist())
    cat_maps[c] = {v: i for i, v in enumerate(u)}
    df[c + '_enc'] = df[c].map(cat_maps[c]).fillna(0).astype('int32')
FEATS = NUM + [c + '_enc' for c in CAT]

# split by episode (holdout)
eps = df['episode_id'].drop_duplicates().sample(frac=1.0, random_state=0).tolist()
cut = int(len(eps) * 0.85)
tr_ep = set(eps[:cut]); va_ep = set(eps[cut:])
tr = df[df['episode_id'].isin(tr_ep)].sort_values('decision_id')
va = df[df['episode_id'].isin(va_ep)].sort_values('decision_id')

def grp(d): return d.groupby('decision_id', sort=True).size().values
# winner-weight 4x like Alakazam
wtr = np.where(tr['won'].fillna(0).astype(float) > 0, 4.0, 1.0)

dtr = lgb.Dataset(tr[FEATS].values, label=tr['is_chosen'].astype(int).values,
                  group=grp(tr), weight=wtr)
dva = lgb.Dataset(va[FEATS].values, label=va['is_chosen'].astype(int).values,
                  group=grp(va), reference=dtr)
params = {'objective': 'lambdarank', 'metric': 'ndcg', 'ndcg_eval_at': [1],
          'num_leaves': 63, 'learning_rate': 0.05, 'lambdarank_truncation_level': 1,
          'min_child_samples': 10, 'feature_fraction': 0.9, 'verbose': -1}
bst = lgb.train(params, dtr, num_boost_round=300, valid_sets=[dva],
                callbacks=[lgb.early_stopping(40, verbose=False)])

# holdout top1
va = va.copy(); va['pred'] = bst.predict(va[FEATS].values)
def top1(d):
    hit = tot = 0
    for did, g in d.groupby('decision_id'):
        if len(g) < 1: continue
        pi = g['pred'].values.argmax()
        chosen = g['is_chosen'].values
        hit += int(chosen[pi] == 1); tot += 1
    return hit / max(1, tot), tot
acc, n = top1(va)
print(f'{ARCH} imitation LGBM holdout top1 = {acc:.4f} over {n} decisions, best_iter={bst.best_iteration}')
imp = sorted(zip(FEATS, bst.feature_importance()), key=lambda t: -t[1])[:8]
print('top feats:', [(f, int(v)) for f, v in imp])
