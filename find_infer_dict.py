import json
nb = json.load(open('/kaggle/working/pokemon-20260627-v0-08d19.ipynb'))
src = ''.join(nb['cells'][19]['source'])
# Find the exact string around prize_gap in inference row dict
idx = src.find("'prize_gap': _prize_gap")
print(f"idx={idx}")
print(repr(src[idx:idx+300]))
