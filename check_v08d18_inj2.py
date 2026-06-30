import json
nb = json.load(open('/kaggle/working/pokemon-20260627-v0-08d18.ipynb'))
src19 = ''.join(nb['cells'][19]['source'])

# Full injection context around _prize_gap
idx = 58867
print("context around _prize_gap (58867):")
print(repr(src19[max(0,idx-600):idx+200]))
print()
# Also show the end where prize_gap is added to dict
idx2 = 62159
print("context around prize_gap dict (62159):")
print(repr(src19[max(0,idx2-100):idx2+200]))
