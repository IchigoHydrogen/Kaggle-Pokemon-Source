import json
nb = json.load(open('/kaggle/working/pokemon-20260627-v0-08d19.ipynb'))
src = ''.join(nb['cells'][19]['source'])
# Show context around inference dict at pos 65170
print("Context around pos 65170:")
print(repr(src[65100:65400]))
