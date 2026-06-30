import json
nb = json.load(open('/kaggle/working/pokemon-20260627-v0-08d18.ipynb'))
src19 = ''.join(nb['cells'][19]['source'])

# Search for 'turn' in injection block  
# injection block is a string (with \\n), so search for turn\\' or \\'turn or turn\\\\n
for pat in ["turn\\'", "\\'turn", "obs.current.turn", "current_turn", ".turn"]:
    idx = src19.find(pat)
    while idx != -1:
        ctx = src19[max(0,idx-50):idx+80]
        print(f"'{pat}' at {idx}:", repr(ctx))
        idx = src19.find(pat, idx+1)
    print()
