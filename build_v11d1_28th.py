"""Patch the v08d34 notebook to mine+train the LGBM on the 20260628 dataset.

Produces a fresh 28th-data rule+LGBM submission (same v08d34 recipe: lambdarank +
trunc1 + features). Afterwards, build_v09d4_submission.py injects forward-sim +
terminal awareness -> v11d1 on 20260628 data.

Changes vs v08d34 notebook:
  - episode root -> 2026-06-28 ; top200 ranking -> 2026-06-28
  - RUN_PREFIX -> pokemon-20260628-v0-base28
  - disable preload (no 28th cache -> mine fresh from 28th episodes)
"""
import json

SRC = '/kaggle/working/pokemon-20260627-v0-08d34.ipynb'
DST = '/kaggle/working/pokemon-20260628-v0-base28.ipynb'
with open(SRC) as f:
    nb = json.load(f)
cells = nb['cells']

# cell1
s1 = ''.join(cells[1]['source'])
assert "episodes-2026-06-26'" in s1
s1 = s1.replace("pokemon-tcg-ai-battle-episodes-2026-06-26'",
                "pokemon-tcg-ai-battle-episodes-2026-06-28'")
assert "top200-20260626-ranking/top200-20260626-ranking.csv" in s1
s1 = s1.replace("top200-20260626-ranking/top200-20260626-ranking.csv",
                "top200-20260628-ranking/top200-20260628-ranking.csv")
assert "RUN_PREFIX = 'pokemon-20260627-v0-08d34'" in s1
s1 = s1.replace("RUN_PREFIX = 'pokemon-20260627-v0-08d34'",
                "RUN_PREFIX = 'pokemon-20260628-v0-base28'")
cells[1]['source'] = s1.splitlines(keepends=True)

# cell7: disable preload (force fresh mining on 28th)
s7 = ''.join(cells[7]['source'])
old_pl = (
    "_PRELOAD_CANDIDATES = [\n"
    "    Path('/kaggle/working/pokemon-20260627-v0-08d19/pokemon-20260627-v0-08d19'),\n"
    "    Path('/kaggle/working/pokemon-20260627-v0-08d18/pokemon-20260627-v0-08d18'),\n"
    "]\n"
)
assert old_pl in s7, 'preload block not found'
s7 = s7.replace(old_pl, "_PRELOAD_CANDIDATES = []  # 28th: no cache -> mine fresh\n")
cells[7]['source'] = s7.splitlines(keepends=True)

# clear outputs
for c in cells:
    if c.get('cell_type') == 'code':
        c['outputs'] = []; c['execution_count'] = None
    else:
        c.pop('outputs', None); c.pop('execution_count', None)

with open(DST, 'w') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print('Written:', DST)

# sanity
with open(DST) as f:
    nb2 = json.load(f)
s1c = ''.join(nb2['cells'][1]['source']); s7c = ''.join(nb2['cells'][7]['source'])
assert 'episodes-2026-06-28' in s1c
assert 'top200-20260628-ranking' in s1c
assert "RUN_PREFIX = 'pokemon-20260628-v0-base28'" in s1c
assert '_PRELOAD_CANDIDATES = []' in s7c
assert 'episodes-2026-06-26' not in s1c
print('sanity OK')
