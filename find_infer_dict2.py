import json
nb = json.load(open('/kaggle/working/pokemon-20260627-v0-08d19.ipynb'))
src = ''.join(nb['cells'][19]['source'])
# Search in escaped form (inference code is in a string literal)
idx = src.find("prize_gap")
count = 0
while idx >= 0 and count < 20:
    chunk = src[max(0,idx-30):idx+100]
    if 'prize' in chunk.lower() and ('infer' in chunk.lower() or '_prize_gap' in chunk or "row" in chunk.lower()):
        print(f"pos={idx}: {repr(chunk[:120])}")
    idx = src.find("prize_gap", idx+1)
    count += 1
