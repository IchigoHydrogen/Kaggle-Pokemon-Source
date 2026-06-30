import json
nb = json.load(open('/kaggle/working/pokemon-20260627-v0-08d10.ipynb'))
src19 = ''.join(nb['cells'][19]['source'])

# ALAKAZAM_OPTION_MODEL_DFの構築方法
idx = src19.find('ALAKAZAM_OPTION_MODEL_DF =')
print('=== ALAKAZAM_OPTION_MODEL_DF build ===')
print(src19[max(0,idx-200):idx+2000])
