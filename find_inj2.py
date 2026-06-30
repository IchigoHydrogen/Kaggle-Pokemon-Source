import json
nb = json.load(open('/kaggle/working/pokemon-20260627-v0-08d10.ipynb'))
src19 = ''.join(nb['cells'][19]['source'])

# v08d10の注入コードのrow dict終わり部分（\' で検索）
# 注入コード内でopp_archがどこにあるかを正確に見る
idx = src19.find("\\'opponent_archetype_norm\\': _op_arch")
print('=== injection row with opp_arch ===')
print(repr(src19[max(0,idx-200):idx+300]))
