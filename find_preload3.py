import json
nb = json.load(open('/kaggle/working/pokemon-20260627-v0-08d10.ipynb'))
src7 = ''.join(nb['cells'][7]['source'])
idx = src7.find('_PRELOAD_CANDIDATES')
# 前後を丁寧に確認
block = src7[max(0,idx-200):idx+900]
print(repr(block))
