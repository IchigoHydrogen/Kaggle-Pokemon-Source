import json
nb = json.load(open('/kaggle/working/pokemon-20260627-v0-08d10.ipynb'))
src7 = ''.join(nb['cells'][7]['source'])
# preload部分を見る
idx = src7.find('_PRELOAD')
if idx >= 0:
    print('=== preload block ===')
    print(src7[max(0,idx-50):idx+800])
else:
    print('No _PRELOAD found in Cell[7]')
    # RUN_REPLAY_MININGの前後を見る
    idx2 = src7.find('RUN_REPLAY_MINING')
    print(src7[max(0,idx2-100):idx2+100])
