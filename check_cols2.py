import pandas as pd, glob, os
# Check the valid predictions parquet for active_id columns
files = glob.glob('/kaggle/working/pokemon-20260627-v0-08d28/**/*.parquet', recursive=True)
print("all files:", [os.path.basename(f) for f in files])
for f in files:
    try:
        df = pd.read_parquet(f)
        active_cols = [c for c in df.columns if 'active' in c.lower()]
        if active_cols:
            print(f"{os.path.basename(f)}: {active_cols}")
        # Also check first file columns
        if 'valid_pred' in f or 'lgbm' in f:
            print(f"  all cols: {list(df.columns)[:30]}")
    except: pass
