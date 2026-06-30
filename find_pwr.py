import json
nb=json.load(open('/kaggle/working/pokemon-20260627-v0-09d2.ipynb'))
src=''.join(nb['cells'][19]['source'])
i = src.find("v08d28 position_winrate:")
# find the print line containing it
j = src.rfind("\n", 0, i)
k = src.find("\n", i)
print(repr(src[j+1:k+1]))
