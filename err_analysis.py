import pandas as pd
pfx = '/kaggle/working/pokemon-20260627-v0-08d19/pokemon-20260627-v0-08d19'
df = pd.read_parquet(pfx + '-unknown0_mlp_valid_predictions.parquet')

# Compute predicted rank within each decision (using MLP scores)
df['mlp_rank'] = df.groupby('decision_id')['unknown0_mlp_pred'].rank(ascending=False, method='min')

# Get per-decision correctness (chosen option = top-1 ranked by MLP)
dec_df = df[df['is_chosen'] == True].copy()
dec_df['mlp_top1'] = (dec_df['mlp_rank'] == 1)

print(f"Total decisions: {len(dec_df)}")
print(f"MLP top1: {dec_df['mlp_top1'].mean():.4f}")
print()

print("=== Error rate by opponent archetype ===")
arch = dec_df.groupby('opponent_archetype_norm')['mlp_top1'].agg(['mean','count'])
arch.columns = ['top1_acc','decisions']
print(arch.sort_values('top1_acc'))
print()

print("=== Error rate by op_last_context ===")
ctx = dec_df.groupby('op_last_context')['mlp_top1'].agg(['mean','count'])
ctx.columns = ['top1_acc','decisions']
print(ctx.sort_values('top1_acc'))
print()

print("=== Error rate by prize_gap ===")
pg = dec_df.groupby('prize_gap')['mlp_top1'].agg(['mean','count'])
pg.columns = ['top1_acc','decisions']
print(pg.sort_values('prize_gap'))
print()

print("=== option_type in WRONG decisions (mlp_top1==False) vs ALL ===")
wrong = dec_df[~dec_df['mlp_top1']]
all_ot = dec_df['option_type'].value_counts(normalize=True).rename('all_rate')
wrong_ot = wrong['option_type'].value_counts(normalize=True).rename('wrong_rate')
ot_cmp = pd.concat([all_ot, wrong_ot], axis=1).fillna(0)
ot_cmp['error_lift'] = ot_cmp['wrong_rate'] / (ot_cmp['all_rate'] + 1e-9)
print(ot_cmp.sort_values('error_lift', ascending=False).head(10))
