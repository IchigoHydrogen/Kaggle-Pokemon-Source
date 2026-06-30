import json
nb = json.load(open('/kaggle/working/pokemon-20260627-v0-08d10.ipynb'))
src19 = ''.join(nb['cells'][19]['source'])

# ALAKAZAM_OPTION_MODEL_DFを全文検索で追う
lines = src19.split('\n')
for i, l in enumerate(lines):
    if 'ALAKAZAM_OPTION_MODEL_DF' in l or 'option_model' in l.lower():
        print(f'{i}: {l[:120]}')
