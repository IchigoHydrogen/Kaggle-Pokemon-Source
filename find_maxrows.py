import json
nb = json.load(open('/kaggle/working/pokemon-20260627-v0-08d19.ipynb'))
src19 = ''.join(nb['cells'][19]['source'])
for pat in ["max_rows=350_000", "max_rows = 350_000", "350000", "350_000"]:
    idx = src19.find(pat)
    if idx != -1:
        print(f"Found '{pat}' at {idx}:", repr(src19[max(0,idx-30):idx+60]))
