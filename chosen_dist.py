import pandas as pd
pfx = '/kaggle/working/pokemon-20260627-v0-08d19/pokemon-20260627-v0-08d19'

# Look at OPTION_ROWS_DF to understand chosen option_type distribution
opt = pd.read_parquet(pfx + '-option_rows.parquet')
dr = pd.read_parquet(pfx + '-decision_rows.parquet')

# Join decision context with chosen options
chosen = opt[opt['is_chosen'] == True][['decision_id','option_type']].merge(
    dr[['decision_id','context_name']], on='decision_id', how='left')

print("=== Chosen option_type by context (top 5 each) ===")
for ctx in ['UNKNOWN_0','TO_BENCH','ATTACH_FROM','SETUP_BENCH_POKEMON']:
    sub = chosen[chosen['context_name'] == ctx]
    print(f"\n{ctx} (n={len(sub)}):")
    print(sub['option_type'].value_counts().head(8).to_string())
