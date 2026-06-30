import json
nb=json.load(open('/kaggle/working/pokemon-20260627-v0-09d1.ipynb'))
src=''.join(nb['cells'][19]['source'])
i = src.find("        else:\n            work['op_last_context'] = 'NONE'\n            work['steps_since_op'] = -1.0\n        # v08d28: position_winrate")
print('anchor found:', i!=-1)
print(repr(src[i:i+180]))
