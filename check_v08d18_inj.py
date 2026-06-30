import json
nb = json.load(open('/kaggle/working/pokemon-20260627-v0-08d18.ipynb'))
src19 = ''.join(nb['cells'][19]['source'])

# Find the injection code block (it's a Python string with \\n)
# Look for prize_gap in the injection code
idx = src19.find("v08d18: prize_gap\\n")
if idx == -1:
    idx = src19.find("prize_gap\\\\n")
print("idx of v08d18 prize_gap inj:", idx)

# Search for _prize_gap in injection code
idx2 = src19.find("_prize_gap")
while idx2 != -1:
    print(f"  _prize_gap at {idx2}:", repr(src19[max(0,idx2-30):idx2+80]))
    idx2 = src19.find("_prize_gap", idx2+1)
