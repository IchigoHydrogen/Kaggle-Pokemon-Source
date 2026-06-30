import json
nb = json.load(open('/kaggle/working/pokemon-20260627-v0-08d10.ipynb'))
src19 = ''.join(nb['cells'][19]['source'])

# injection code内でoption_signature行の後の部分を探す（row dictの終わり）
idx = src19.find("'option_signature': sig,")
print('=== option_signature row end ===')
print(repr(src19[idx:idx+200]))

# v08d10ではopp_archがここに追加されている
idx2 = src19.find("'opponent_archetype_norm': _op_arch")
print('\n=== opp_arch location ===')
print(repr(src19[max(0,idx2-50):idx2+200]))

# Cell[6]: prizes lines
src6 = ''.join(nb['cells'][6]['source'])
idx3 = src6.find("my_prizes_left")
print('\n=== prizes_left in Cell[6] ===')
print(repr(src6[max(0,idx3-50):idx3+200]))
