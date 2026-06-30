import json
nb = json.load(open('/kaggle/working/pokemon-20260627-v0-08d10.ipynb'))
src19 = ''.join(nb['cells'][19]['source'])

# 注入コードの最初の100行を見る（変数名の確認）
idx = src19.find('_LGBM_INJ_CODE')
if idx >= 0:
    print(src19[idx:idx+3000])
