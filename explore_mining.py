import json
nb = json.load(open('/kaggle/working/pokemon-20260627-v0-08d10.ipynb'))

# state_summary_from_obs の定義を探す
for ci, cell in enumerate(nb['cells']):
    src = ''.join(cell['source'])
    if 'def state_summary_from_obs' in src or 'prizes_left' in src:
        idx = src.find('prizes_left')
        if idx >= 0:
            print(f'=== Cell[{ci}]: prizes_left context ===')
            print(src[max(0,idx-300):idx+600])
            print()
        if 'def state_summary_from_obs' in src:
            idx2 = src.find('def state_summary_from_obs')
            print(f'=== Cell[{ci}]: state_summary_from_obs ===')
            print(src[idx2:idx2+2500])
            break
