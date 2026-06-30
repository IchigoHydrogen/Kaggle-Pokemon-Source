import json

ep_dir = '/kaggle/input/competitions/pokemon-tcg-ai-battle-episodes-2026-06-26'
data = json.load(open(ep_dir + '/81919329.json'))

steps = data['steps']
print(f'steps[0] type: {type(steps[0])}')
s0 = steps[0]
if isinstance(s0, list):
    print(f'steps[0] len: {len(s0)}')
    print(f'steps[0][0] keys: {list(s0[0].keys()) if isinstance(s0[0], dict) else type(s0[0])}')
    # observationを探す
    for i, item in enumerate(s0):
        if isinstance(item, dict):
            obs = item.get('observation', item.get('obs', None))
            if obs:
                print(f'obs found at steps[0][{i}]')
                cur = obs.get('current', {}) if isinstance(obs, dict) else {}
                players = cur.get('players', [])
                print(f'players: {len(players)}')
                if players:
                    p = players[0]
                    print(f'player keys: {list(p.keys())}')
                    print(f'prize: {repr(p.get("prize", "MISSING"))[:200]}')
                    print(f'prizeCount: {p.get("prizeCount", "MISSING")}')
                break

# ゲームの中盤あたりを見る
for si in [20, 40, 60]:
    if si < len(steps):
        step = steps[si]
        if isinstance(step, list):
            for item in step:
                if isinstance(item, dict) and 'observation' in item:
                    obs = item['observation']
                    cur = obs.get('current', {})
                    players = cur.get('players', [])
                    if players and 'prize' in players[0]:
                        p = players[0]
                        print(f'step[{si}] prize: {repr(p["prize"])[:100]} prizeCount: {p.get("prizeCount")}')
                    break
