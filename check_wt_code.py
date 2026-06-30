import json
nb = json.load(open('/kaggle/working/pokemon-20260627-v0-08d19.ipynb'))
src19 = ''.join(nb['cells'][19]['source'])

# Find _winner_weight and surrounding sample weight code
idx = src19.find('_winner_weight = 4.0')
print("=== winner_weight context ===")
print(repr(src19[max(0,idx-100):idx+600]))
