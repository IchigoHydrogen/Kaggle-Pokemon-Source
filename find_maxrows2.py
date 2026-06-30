import json
nb = json.load(open('/kaggle/working/pokemon-20260627-v0-08d19.ipynb'))
# Search all cells for max_rows
for ci, cell in enumerate(nb['cells']):
    src = ''.join(cell['source'])
    for pat in ["max_rows", "MAX_ROWS", "n_rows", "350_000", "350000"]:
        idx = src.find(pat)
        while idx != -1:
            ctx = src[max(0,idx-30):idx+60]
            if "lgbm" in ctx.lower() or "train" in ctx.lower() or "350" in ctx or "rows" in ctx.lower():
                print(f"Cell[{ci}] '{pat}' at {idx}:", repr(ctx))
            idx = src.find(pat, idx+1)
