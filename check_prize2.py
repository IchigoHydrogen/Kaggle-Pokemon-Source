import json

ep_dir = '/kaggle/input/competitions/pokemon-tcg-ai-battle-episodes-2026-06-26'
f = ep_dir + '/81919329.json'
data = json.load(open(f))

# 構造を確認
print('top-level keys:', list(data.keys()) if isinstance(data, dict) else 'LIST len=' + str(len(data)))

if isinstance(data, dict):
    for k in list(data.keys())[:5]:
        print(f'  {k}: {repr(str(data[k])[:100])}')

# stepsを見つける
steps = data.get('steps', data.get('observations', []))
if not steps and isinstance(data, list):
    steps = data

print(f'\nsteps count: {len(steps)}')
if steps:
    s0 = steps[0]
    print('step[0] keys:', list(s0.keys()) if isinstance(s0, dict) else type(s0))
    obs = s0.get('obs', s0) if isinstance(s0, dict) else s0
    cur = obs.get('current', {}) if isinstance(obs, dict) else {}
    players = cur.get('players', [])
    print(f'players count: {len(players)}')
    if players:
        p0 = players[0]
        print(f'player[0] keys: {list(p0.keys())}')
        print(f'prize: {repr(p0.get("prize", "MISSING"))[:200]}')
        print(f'prizeCount: {repr(p0.get("prizeCount", "MISSING"))}')
        print(f'prizes: {repr(p0.get("prizes", "MISSING"))[:200]}')
