import json
nb = json.load(open('/kaggle/working/pokemon-20260627-v0-08d10.ipynb'))
src7 = ''.join(nb['cells'][7]['source'])
idx = src7.find('_PRELOAD_CANDIDATES')
print(repr(src7[idx:idx+1200]))
