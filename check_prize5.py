import json

ep_dir = '/kaggle/input/competitions/pokemon-tcg-ai-battle-episodes-2026-06-26'
data = json.load(open(ep_dir + '/81919329.json'))
steps = data['steps']

# prize listの長さが変わるstepを探す
prev_prize_len = {0: None, 1: None}
for si in range(len(steps)):
    step = steps[si]
    if not isinstance(step, list):
        continue
    for player_obs_i, item in enumerate(step):
        if not isinstance(item, dict):
            continue
        obs = item.get('observation')
        if not isinstance(obs, dict):
            continue
        cur = obs.get('current')
        if not isinstance(cur, dict):
            continue
        players = cur.get('players', [])
        if not isinstance(players, list) or len(players) < 2:
            continue
        
        for pi in range(2):
            if pi >= len(players) or not isinstance(players[pi], dict):
                continue
            prize = players[pi].get('prize', [])
            prize_len = len(prize) if isinstance(prize, list) else -1
            key = (player_obs_i, pi)
            if key not in prev_prize_len:
                prev_prize_len[key] = prize_len
            elif prev_prize_len[key] != prize_len:
                print(f'step[{si}][obs={player_obs_i}] player[{pi}] prize changed: {prev_prize_len[key]} -> {prize_len}')
                prev_prize_len[key] = prize_len

# 最後のstepのprize状況
for si in [-10, -5, -3, -1]:
    step = steps[si]
    if isinstance(step, list) and step:
        item = step[0]
        if isinstance(item, dict):
            obs = item.get('observation', {})
            cur = obs.get('current', {}) if isinstance(obs, dict) else {}
            players = cur.get('players', []) if isinstance(cur, dict) else []
            if players:
                print(f'step[{si}]: p0 prize={len(players[0].get("prize",[]))} p1 prize={len(players[1].get("prize",[]) if len(players)>1 else [])}')
