import pandas as pd, glob, os
# Look at the parquet files from v08d28 to see what columns are available
files = glob.glob('/kaggle/working/pokemon-20260627-v0-08d28/**/*.parquet', recursive=True)
print("parquet files:", files[:5])
for f in files[:3]:
    try:
        df = pd.read_parquet(f)
        cols = [c for c in df.columns if 'active' in c.lower() or 'my_' in c.lower()]
        print(f"{os.path.basename(f)}: active/my cols = {cols[:20]}")
    except: pass
