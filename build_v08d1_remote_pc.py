"""Build pokemon-20260627-v0-08d1-remote-pc.ipynb from v05d25 notebook.

v08d1: top200 expert data + LGBM LambdaRank (v05d25 proven approach).

Core hypothesis:
  v05d25 (869.7) was trained on 20260624 rule-agent episodes.
  v08d1 uses 20260626 episodes where top200 agents (rate 988-1401) participated.
  Same LGBM LambdaRank objective + winner_weight=4x, but far better training data.

Changes from v05d25:
  1. EPISODE_ROOT_CANDIDATES → pokemon-tcg-ai-battle-episodes-2026-06-26 only
  2. TOP200_CSV_PATH → top200-20260626-ranking.csv (latest ranking)
  3. EXPERIMENT_NAME → v0_08d1_top200_lgbm_26
  4. RUN_PREFIX → pokemon-20260627-v0-08d1

Everything else identical to v05d25:
  - TOP200_EPISODE_FILTER = True (only top200-participant episodes)
  - LGBM LambdaRank (objective=lambdarank, ndcg@1,3)
  - winner_weight = 4x
  - Same feature set, same submission packaging
"""
import json

SRC = '/kaggle/working/pokemon-20260625-v0-05d25.ipynb'
DST = '/kaggle/working/pokemon-20260627-v0-08d1.ipynb'

with open(SRC) as f:
    nb = json.load(f)

cells = nb['cells']

# ── Cell[1]: Config ───────────────────────────────────────────────────────────
src1 = ''.join(cells[1]['source'])

# Patch 1: EXPERIMENT_NAME
OLD_EXP = "EXPERIMENT_NAME = 'v0_05d25_lgbm_winner_wt4'"
NEW_EXP = "EXPERIMENT_NAME = 'v0_08d1_top200_lgbm_26'"
assert OLD_EXP in src1, f'EXPERIMENT_NAME not found'
src1 = src1.replace(OLD_EXP, NEW_EXP)

# Patch 2: TOP200_CSV_PATH
OLD_TOP = "TOP200_CSV_PATH = Path('/kaggle/input/competitions/top200-20260624-ranking/top200-20260624-ranking.csv')"
NEW_TOP = "TOP200_CSV_PATH = Path('/kaggle/input/competitions/top200-20260626-ranking/top200-20260626-ranking.csv')"
assert OLD_TOP in src1, f'TOP200_CSV_PATH not found'
src1 = src1.replace(OLD_TOP, NEW_TOP)

# Patch 3: EPISODE_ROOT_CANDIDATES (only 20260626)
OLD_EPS = (
    "EPISODE_ROOT_CANDIDATES = [\n"
    "    Path('/kaggle/input/competitions/pokemon-tcg-ai-battle-episodes-2026-06-24'),\n"
    "]"
)
NEW_EPS = (
    "EPISODE_ROOT_CANDIDATES = [\n"
    "    Path('/kaggle/input/competitions/pokemon-tcg-ai-battle-episodes-2026-06-26'),\n"
    "]"
)
assert OLD_EPS in src1, f'EPISODE_ROOT_CANDIDATES not found'
src1 = src1.replace(OLD_EPS, NEW_EPS)

# Patch 4: RUN_PREFIX
OLD_PFX = "RUN_PREFIX = 'pokemon-20260625-v0-05d25'"
NEW_PFX = "RUN_PREFIX = 'pokemon-20260627-v0-08d1'"
assert OLD_PFX in src1, f'RUN_PREFIX not found'
src1 = src1.replace(OLD_PFX, NEW_PFX)

cells[1]['source'] = src1.splitlines(keepends=True)

# ── Clear outputs ─────────────────────────────────────────────────────────────
for c in cells:
    if c.get('cell_type') == 'code':
        c['outputs'] = []
        c['execution_count'] = None
    else:
        c.pop('outputs', None)
        c.pop('execution_count', None)

with open(DST, 'w') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f'Written: {DST}')

# ── Sanity checks ─────────────────────────────────────────────────────────────
with open(DST) as f:
    nb2 = json.load(f)
s1 = ''.join(nb2['cells'][1]['source'])

assert 'v0_08d1_top200_lgbm_26' in s1,                'EXPERIMENT_NAME'
assert 'top200-20260626-ranking' in s1,                'TOP200_CSV_PATH 2026-06-26'
assert 'top200-20260624-ranking' not in s1,            'no old TOP200_CSV'
assert 'episodes-2026-06-26' in s1,                    'EPISODE_ROOT 2026-06-26'
assert 'episodes-2026-06-24' not in s1,                'no old episode dir'
assert "RUN_PREFIX = 'pokemon-20260627-v0-08d1'" in s1, 'RUN_PREFIX'
assert 'TOP200_EPISODE_FILTER' in s1,                  'TOP200_EPISODE_FILTER kept'
assert 'winner_wt4' not in s1,                         'old experiment name gone'

# LambdaRank kept in the notebook
full = ''.join(''.join(c['source']) for c in nb2['cells'])
assert 'lambdarank' in full,       'lambdarank objective kept'
assert 'winner_weight' in full,    'winner_weight kept'
assert 'ndcg' in full,             'ndcg metric kept'
print('All sanity checks passed.')
