import json
nb = json.load(open('/kaggle/working/pokemon-20260627-v0-08d19.ipynb'))
src19 = ''.join(nb['cells'][19]['source'])
for pat in ["winner_weight", "wt4", "wt_4", "winner_wt", "4x"]:
    idx = src19.find(pat)
    if idx != -1:
        print(f"'{pat}' at {idx}:", repr(src19[max(0,idx-30):idx+60]))
