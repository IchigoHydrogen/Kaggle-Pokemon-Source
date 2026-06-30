import json
nb = json.load(open('/kaggle/working/pokemon-20260627-v0-08d19.ipynb'))
src = ''.join(nb['cells'][19]['source'])
# Find the block after op_last_context computed and steps_since_op
idx = src.find("work['op_last_context'] = work['op_last_context'].fillna")
if idx < 0:
    idx = src.find("v08d19 steps_since_op")
print(f"idx={idx}")
print(repr(src[max(0,idx-50):idx+500]))
