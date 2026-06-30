import json

# v08d10のnotebook出力から特徴量重要度を全部取り出す
nb = json.load(open('/kaggle/working/pokemon-20260627-v0-08d10.ipynb'))

# outputs を探す
for ci, cell in enumerate(nb['cells']):
    for out in cell.get('outputs', []):
        text = ''
        if out.get('output_type') == 'stream':
            text = ''.join(out.get('text', []))
        elif out.get('output_type') == 'execute_result':
            text = ''.join(out.get('data', {}).get('text/plain', []))
        if 'feature_importance' in text or 'lgbm report' in text.lower():
            print(f'=== Cell[{ci}] output ===')
            print(text[:3000])
            print()
