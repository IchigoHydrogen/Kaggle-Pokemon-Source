import json
nb = json.load(open('/kaggle/working/pokemon-20260627-v0-08d10.ipynb'))

# Cell[9] を全部表示 - prepare_alakazam_option_dataset の中身
src9 = ''.join(nb['cells'][9]['source'])
# prepare_alakazam_option_dataset の定義を探す（Cell[9]の前のセルにあるかも）
for ci, cell in enumerate(nb['cells']):
    src = ''.join(cell['source'])
    if 'def prepare_alakazam_option_dataset' in src:
        idx = src.find('def prepare_alakazam_option_dataset')
        print(f'=== Cell[{ci}]: prepare_alakazam_option_dataset ===')
        print(src[idx:idx+3000])
        break
