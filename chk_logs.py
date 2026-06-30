import pandas as pd, glob
# DECISION_ROWS_DF and option model df columns
for pat in ['*v0-09d2*decision_rows*.parquet','*v0-09d2*alakazam_option_model_df*.parquet','*v0-09d2*unknown0*valid*.parquet']:
    fs = glob.glob('/kaggle/working/pokemon-20260627-v0-09d2/'+pat) + glob.glob('/kaggle/working/'+pat)
    if not fs:
        print(pat, '-> none'); continue
    df = pd.read_parquet(fs[0])
    logcols = [c for c in df.columns if 'log' in c.lower() or 'context' in c.lower() or 'event' in c.lower()]
    print(fs[0].split('/')[-1], '| log/context/event cols:', logcols)
