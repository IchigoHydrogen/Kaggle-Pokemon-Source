import json
nb = json.load(open('/kaggle/working/pokemon-20260627-v0-08d18.ipynb'))
src19 = ''.join(nb['cells'][19]['source'])

# Find how 'turn' is used in the injection code
idx = src19.find("'turn'")
while idx != -1:
    ctx = src19[max(0,idx-50):idx+100]
    if "\\n" in ctx:  # inside injection code block
        print(f"at {idx}:", repr(ctx))
    idx = src19.find("'turn'", idx+1)

# Also find the my_ps/op_ps extraction in injection code  
idx2 = src19.find("my_ps")
print("\nmy_ps first occurrence:", repr(src19[max(0,idx2-20):idx2+80]))
