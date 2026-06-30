import json
nb = json.load(open('/kaggle/working/pokemon-20260627-v0-08d23.ipynb'))
s19 = ''.join(nb['cells'][19]['source'])

# Find all occurrences of 'Winner-weighted'
idx = 0
while True:
    idx = s19.find('Winner-weighted', idx)
    if idx == -1:
        break
    print(f"pos={idx}:")
    print(repr(s19[max(0,idx-50):idx+200]))
    print()
    idx += 1
