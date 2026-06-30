import json
nb=json.load(open('/kaggle/working/pokemon-20260627-v0-09d1.ipynb'))
src=''.join(nb['cells'][19]['source'])
i = src.find("work['op_last_context']")
while i != -1:
    print('--- @', i, '---')
    print(src[i-200:i+250])
    print()
    i = src.find("work['op_last_context']", i+1)
