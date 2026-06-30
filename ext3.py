import json
nb=json.load(open('/kaggle/working/pokemon-20260627-v0-09d3.ipynb'))
for cell in nb['cells']:
    if cell.get('cell_type')!='code': continue
    for out in cell.get('outputs',[]):
        txt=''.join(out.get('text', out.get('data',{}).get('text/plain',[])))
        for line in txt.splitlines():
            if 'INFER-FAITHFUL' in line:
                print('>>',line.strip())
