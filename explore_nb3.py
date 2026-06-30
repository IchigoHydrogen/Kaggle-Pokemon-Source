import json
nb = json.load(open('/kaggle/working/pokemon-20260627-v0-08d10.ipynb'))
src19 = ''.join(nb['cells'][19]['source'])

# state_summaryとoption_rowsのjoin部分を探す
for keyword in ['state_summary', 'STATE_SUMMARY', 'pd.merge', '_merge', 'join(']:
    idx = 0
    while True:
        idx = src19.find(keyword, idx)
        if idx == -1:
            break
        print(f'[{idx}] {repr(src19[max(0,idx-30):idx+120])}')
        idx += 1
        if idx > 200000:
            break
