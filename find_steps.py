import json
nb = json.load(open('/kaggle/working/pokemon-20260627-v0-08d19.ipynb'))
src19 = ''.join(nb['cells'][19]['source'])
for pat in ["steps_since_op", "_steps_since_op"]:
    idx = 0
    while True:
        idx = src19.find(pat, idx)
        if idx == -1: break
        print(f"'{pat}' at {idx}:", repr(src19[max(0,idx-30):idx+60]))
        idx += 1
