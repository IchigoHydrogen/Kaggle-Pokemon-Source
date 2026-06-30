"""v13d Step1 (measure-first): is the opponent 'clock' a real, predictive signal?

Clock = how many turns until a player's strongest attack comes online (energy
buildup). For each MY decision in self-play, compute opp_clock and my_clock
(turns to afford the highest-damage attack across that player's Pokemon), and
race_margin = opp_clock - my_clock. Then check win-rate by race_margin / opp_clock.
If it predicts the outcome, the clock is a real signal to use for OFFENSE/tempo.
"""
import sys, importlib.util
from collections import defaultdict
sys.path.insert(0, '/kaggle/input/competitions/pokemon-tcg-ai-battle/sample_submission')
sys.path.insert(0, '/tmp/agent_v11d1_28thfull')
import cg.game as game
from cg.api import all_card_data, all_attack
spec = importlib.util.spec_from_file_location('a', '/tmp/agent_v11d1_28thfull/main.py')
A = importlib.util.module_from_spec(spec); sys.modules['a'] = A; spec.loader.exec_module(A)
deck = A.my_deck

CT = {c.cardId: c for c in all_card_data()}
ATK = {a.attackId: (int(getattr(a, 'damage', 0) or 0), len(getattr(a, 'energies', None) or [])) for a in all_attack()}

def fv(o):
    sel = o.get('select')
    if sel is None: return None
    mn = int(sel.get('minCount', 1) or 1); n = len(sel.get('option', []) or [])
    return list(range(min(max(1, mn), n))) or [0]

def best_attacker_clock(pl):
    """Min turns until this player's HIGHEST-DAMAGE attack is affordable (energy buildup)."""
    best = None  # (clock, damage)
    poks = [p for p in (pl.get('active') or []) if p] + [p for p in (pl.get('bench') or []) if p]
    for pk in poks:
        e = len(pk.get('energies') or [])
        cd = CT.get(pk.get('id'))
        if not cd: continue
        for aid in (getattr(cd, 'attacks', None) or []):
            dmg, cost = ATK.get(aid, (0, 99))
            if dmg <= 0: continue
            clock = max(0, cost - e)  # ~1 energy/turn
            if best is None or dmg > best[1] or (dmg == best[1] and clock < best[0]):
                best = (clock, dmg)
    return best[0] if best else 9, (best[1] if best else 0)

def play_collect(rows):
    A.pre_turn = -1
    obs, sd = game.battle_start(deck, deck)
    if obs is None: return None
    local = []  # (p, opp_clock, my_clock, opp_dmg)
    for _ in range(2000):
        cur = obs.get('current')
        if cur is not None and cur.get('result', -1) != -1:
            w = cur['result']
            for (p, oc, mc, od) in local:
                rows.append((oc, mc, oc - mc, 1 if w == p else 0))
            return w
        sel = obs.get('select')
        if sel is None: return None
        p = cur['yourIndex'] if cur else 0
        if cur and sel.get('context') == 0:
            pls = cur['players']
            mc, _ = best_attacker_clock(pls[p])
            oc, od = best_attacker_clock(pls[1 - p])
            local.append((p, oc, mc, od))
        ch = A.agent(obs) or fv(obs)
        try: obs = game.battle_select(ch)
        except Exception: return None
    return None

rows = []
N = int(sys.argv[1]) if len(sys.argv) > 1 else 60
for g in range(N):
    play_collect(rows); game.battle_finish()

import statistics
print(f'decisions sampled: {len(rows)} from {N} games')
# win rate by race_margin bucket (opp_clock - my_clock)
buckets = defaultdict(lambda: [0, 0])
for oc, mc, rm, won in rows:
    b = 'opp_slower(+,good)' if rm >= 2 else ('even' if -1 <= rm <= 1 else 'opp_faster(-,bad)')
    buckets[b][0] += won; buckets[b][1] += 1
print('--- win-rate by race_margin (opp_clock - my_clock) ---')
for b in ['opp_slower(+,good)', 'even', 'opp_faster(-,bad)']:
    w, n = buckets[b]
    print(f'  {b:20}: n={n:5} winrate={w/max(1,n):.3f}')
# win rate by opp_clock (how soon opp strong attack online)
ob = defaultdict(lambda: [0, 0])
for oc, mc, rm, won in rows:
    k = 'opp_online(0-1)' if oc <= 1 else ('opp_2-3' if oc <= 3 else 'opp_far(4+)')
    ob[k][0] += won; ob[k][1] += 1
print('--- win-rate by opp_clock (turns to opp strong attack) ---')
for k in ['opp_online(0-1)', 'opp_2-3', 'opp_far(4+)']:
    w, n = ob[k]
    print(f'  {k:16}: n={n:5} winrate={w/max(1,n):.3f}')
