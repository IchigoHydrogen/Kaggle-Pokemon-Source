import pandas as pd
pfx = '/kaggle/working/pokemon-20260627-v0-08d19/pokemon-20260627-v0-08d19'
df = pd.read_parquet(pfx + '-unknown0_mlp_valid_predictions.parquet')

# Check what option_type 12 and 14 are in TO_BENCH context
to_bench = df[df['op_last_context'] == 'TO_BENCH']
for ot in [12, 14]:
    sub = to_bench[to_bench['option_type'] == ot]
    print(f"=== option_type={ot} in TO_BENCH (n={len(sub)}) ===")
    print("option_signature dist:")
    print(sub['option_signature'].value_counts().head(5))
    print("num_options dist:")
    print(sub['num_options'].value_counts().head(5))
    print("chosen examples:")
    chosen = sub[sub['is_chosen']==True]
    print(chosen[['option_index','option_type','option_signature','num_options','in_play_area']].head(5).to_string())
    print()

# Also check what my_alakazam_count etc. are -- are they in features?
import json
nb = json.load(open('/kaggle/working/pokemon-20260627-v0-08d19.ipynb'))
src = ''.join(nb['cells'][19]['source'])
idx = src.find('UNKNOWN0_NUMERIC_FEATURES')
print("=== NUMERIC_FEATURES ===")
print(src[idx:idx+600])
