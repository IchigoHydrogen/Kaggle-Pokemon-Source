import pandas as pd
pfx = '/kaggle/working/pokemon-20260627-v0-08d19/pokemon-20260627-v0-08d19'
df = pd.read_parquet(pfx + '-unknown0_mlp_valid_predictions.parquet')
df['mlp_rank'] = df.groupby('decision_id')['unknown0_mlp_pred'].rank(ascending=False, method='min')
dec_df = df[df['is_chosen'] == True].copy()
dec_df['mlp_top1'] = (dec_df['mlp_rank'] == 1)

# TO_BENCH cases: 2807 decisions, top1=0.524 - most impactful failure mode
to_bench = dec_df[dec_df['op_last_context'] == 'TO_BENCH']
unk0 = dec_df[dec_df['op_last_context'] == 'UNKNOWN_0']
print(f"TO_BENCH decisions: {len(to_bench)}, top1={to_bench['mlp_top1'].mean():.4f}")
print(f"UNKNOWN_0 decisions: {len(unk0)}, top1={unk0['mlp_top1'].mean():.4f}")
print()

# In TO_BENCH cases, what option_types am I choosing?
print("=== option_type dist in TO_BENCH cases (chosen only) ===")
print(to_bench['option_type'].value_counts().head(10))
print()

# In TO_BENCH, what's my error by option_type?
print("=== top1 by option_type in TO_BENCH cases ===")
tb_ot = to_bench.groupby('option_type')['mlp_top1'].agg(['mean','count'])
print(tb_ot.sort_values('mean'))
print()

# In UNKNOWN_0 op_last_ctx, what is the actual game phase?
print("=== turn_bucket when op_last_context=TO_BENCH vs UNKNOWN_0 ===")
print("TO_BENCH:")
print(to_bench['unknown0_turn_bucket'].value_counts())
print("UNKNOWN_0:")
print(unk0['unknown0_turn_bucket'].value_counts())
print()

# Prize gap distribution for each op_last_context
print("=== prize_gap means ===")
print(dec_df.groupby('op_last_context')[['prize_gap','mlp_top1']].agg({'prize_gap':'mean','mlp_top1':'mean'}).sort_values('mlp_top1'))
