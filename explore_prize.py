import json
nb = json.load(open('/kaggle/working/pokemon-20260627-v0-08d10.ipynb'))

# count_zone関数の定義を探す
for ci, cell in enumerate(nb['cells']):
    src = ''.join(cell['source'])
    if 'def count_zone' in src:
        idx = src.find('def count_zone')
        print(f'=== Cell[{ci}]: count_zone ===')
        print(src[idx:idx+1000])
        print()

# player_state_from_obs の定義を探す（prize zone の取り方）
for ci, cell in enumerate(nb['cells']):
    src = ''.join(cell['source'])
    if 'def player_state_from_obs' in src:
        idx = src.find('def player_state_from_obs')
        print(f'=== Cell[{ci}]: player_state_from_obs ===')
        print(src[idx:idx+1500])
        print()
