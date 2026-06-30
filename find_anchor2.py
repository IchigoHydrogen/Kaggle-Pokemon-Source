import json
nb = json.load(open('/kaggle/working/pokemon-20260627-v0-08d19.ipynb'))
src = ''.join(nb['cells'][19]['source'])
# Find the section just before _extra_numeric_candidates
idx = src.find("_extra_numeric_candidates")
print(repr(src[max(0,idx-400):idx+300]))
