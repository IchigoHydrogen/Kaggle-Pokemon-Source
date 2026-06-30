import json
nb = json.load(open('/kaggle/working/pokemon-20260627-v0-08d19.ipynb'))
src19 = ''.join(nb['cells'][19]['source'])

idx = src19.find('_winner_weight = 4.0')
# Show 50 chars before and 800 chars after
chunk = src19[max(0,idx-200):idx+800]
# Print with visible newlines for exact matching
for i, line in enumerate(chunk.split('\n')):
    print(f'{i:3d}| {repr(line)}')
