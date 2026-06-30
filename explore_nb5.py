import json
nb = json.load(open('/kaggle/working/pokemon-20260627-v0-08d10.ipynb'))

# Cell[19]以外のセルでALAKAZAM_OPTION_MODEL_DFを探す
for ci, cell in enumerate(nb['cells']):
    src = ''.join(cell['source'])
    if 'ALAKAZAM_OPTION_MODEL_DF' in src and ci != 19:
        print(f'=== Cell[{ci}] ===')
        # 代入行だけ抜き出す
        for l in src.split('\n'):
            if 'ALAKAZAM_OPTION_MODEL_DF' in l:
                print(l[:200])
        print()
