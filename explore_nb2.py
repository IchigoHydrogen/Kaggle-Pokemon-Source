import json
nb = json.load(open('/kaggle/working/pokemon-20260627-v0-08d10.ipynb'))
src19 = ''.join(nb['cells'][19]['source'])

# ALAKAZAM_OPTION_MODEL_DFの実際の代入を探す（小文字含む）
for keyword in ['ALAKAZAM_OPTION_MODEL_DF', 'option_model_df', '_build_option_model', 'state_summary', 'merge(']:
    idx = src19.find(keyword + ' =')
    if idx == -1:
        idx = src19.find(keyword + '=')
    if idx >= 0:
        print(f'=== Found: {keyword} at {idx} ===')
        print(src19[max(0,idx-50):idx+500])
        print()

# LGBM feature importance出力を探す
idx = src19.find('feature_importance')
if idx >= 0:
    print('=== feature_importance ===')
    print(src19[max(0,idx-100):idx+800])
