import json, os, glob

# episodeファイルを1つ読んで prize の構造を確認
ep_dir = '/kaggle/input/competitions/pokemon-tcg-ai-battle-episodes-2026-06-26/episodes'
files = glob.glob(ep_dir + '/*.json')[:3]

for f in files:
    data = json.load(open(f))
    # episodes配列の最初のstepを見る
    episodes = data if isinstance(data, list) else [data]
    for ep in episodes[:1]:
        steps = ep.get('steps', ep.get('observations', []))
        for step in steps[:5]:
            obs = step.get('obs', step)
            cur = obs.get('current', {})
            players = cur.get('players', [])
            if players:
                p = players[0]
                prize = p.get('prize', 'MISSING')
                prizeCount = p.get('prizeCount', 'MISSING')
                print(f'prize field: {repr(prize)[:100]}')
                print(f'prizeCount field: {repr(prizeCount)}')
                # 他のフィールドも確認
                print(f'player keys: {list(p.keys())[:20]}')
                print()
                break
        break
    break
