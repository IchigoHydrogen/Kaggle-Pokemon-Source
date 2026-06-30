import json

ep_dir = '/kaggle/input/competitions/pokemon-tcg-ai-battle-episodes-2026-06-26'
data = json.load(open(ep_dir + '/81919329.json'))
steps = data['steps']

# stepを1つずつ走査してprize情報を持つものを探す
for si in range(0, min(100, len(steps))):
    step = steps[si]
    if not isinstance(step, list):
        continue
    for player_i, item in enumerate(step):
        if not isinstance(item, dict):
            continue
        obs = item.get('observation')
        if obs is None or not isinstance(obs, dict):
            continue
        cur = obs.get('current')
        if not isinstance(cur, dict):
            continue
        players = cur.get('players', [])
        if not isinstance(players, list) or len(players) == 0:
            continue
        p = players[0]
        if not isinstance(p, dict):
            continue
        prize_val = p.get('prize', 'MISSING')
        prize_count = p.get('prizeCount', 'MISSING')
        if prize_val != 'MISSING' or prize_count != 'MISSING':
            print(f'step[{si}][{player_i}]: prize={repr(prize_val)[:80]}, prizeCount={prize_count}')
            # 全プレイヤーのprize
            for pi2, p2 in enumerate(players):
                if isinstance(p2, dict):
                    print(f'  player[{pi2}] prize={repr(p2.get("prize","?"))[:60]} prizeCount={p2.get("prizeCount","?")}')
            if si > 10:
                break
    if si > 30:
        break
