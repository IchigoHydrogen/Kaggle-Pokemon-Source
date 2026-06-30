import json
nb = json.load(open('/kaggle/working/pokemon-20260627-v0-08d19.ipynb'))
src19 = ''.join(nb['cells'][19]['source'])
idx = src19.find("num_leaves")
print(repr(src19[max(0,idx-50):idx+80]))
# Also check max_rows
idx2 = src19.find("max_rows")
print(repr(src19[max(0,idx2-50):idx2+80]))
