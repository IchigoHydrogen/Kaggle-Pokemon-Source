import json
nb = json.load(open('/kaggle/working/pokemon-20260627-v0-08d28.ipynb'))
# Find what columns are available in work after it's built
for i, cell in enumerate(nb['cells']):
    if cell.get('cell_type') != 'code': continue
    for out in cell.get('outputs', []):
        txt = ''.join(out.get('text', out.get('data', {}).get('text/plain', [])))
        if 'my_active' in txt or 'active_id' in txt or 'columns' in txt.lower():
            print(f'cell[{i}]: {txt[:300]}')
