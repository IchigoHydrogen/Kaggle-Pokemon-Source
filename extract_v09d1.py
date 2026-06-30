import json
nb = json.load(open('/kaggle/working/pokemon-20260627-v0-09d1.ipynb'))
for cell in nb['cells']:
    if cell.get('cell_type') != 'code': continue
    for out in cell.get('outputs', []):
        txt = ''.join(out.get('text', out.get('data', {}).get('text/plain', [])))
        for line in txt.splitlines():
            if 'INFER-FAITHFUL' in line:
                print('>>', line.strip())
        if 'unknown0 lgbm report' in txt:
            i = txt.find("'unknown0_lgbm_decision_metrics_infer_faithful'")
            j = txt.find("'unknown0_lgbm_decision_metrics'")
            print('-- real metrics:', txt[j:j+120].replace(chr(10),' '))
            if i>0: print('-- infer-faithful in report:', txt[i:i+160].replace(chr(10),' '))
