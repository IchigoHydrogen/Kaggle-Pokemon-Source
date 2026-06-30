"""v10d foundation: reconstruct a representative LEGAL deck for an opponent archetype
from the battle logs, to build a DIVERSE (non-mirror) validation opponent.

For each episode where a player plays the target archetype, read THAT player's OWN
observations (their hand is visible from their own perspective) and aggregate all
deck cards they reveal across the game (hand + active/bench incl. attached energy,
tools, pre-evolutions + discard + prize). Estimate copies via max simultaneous count.
Build a legal 60-card deck using card metadata (BASIC_ENERGY unlimited, others <=4,
ACE SPEC <=1, >=1 basic Pokemon). Save JSON and validate via engine battle_start.
"""
import sys, json, glob, os
from collections import Counter, defaultdict
sys.path.insert(0, '/kaggle/input/competitions/pokemon-tcg-ai-battle/sample_submission')
import pandas as pd
from cg.api import all_card_data, CardType
import cg.game as game

ARCH = sys.argv[1] if len(sys.argv) > 1 else 'Hop_Trevenant'
N_EP = int(sys.argv[2]) if len(sys.argv) > 2 else 80

ct = {c.cardId: c for c in all_card_data()}
EP_DIRS = ['/kaggle/input/competitions/pokemon-tcg-ai-battle-episodes-2026-06-26',
           '/kaggle/input/competitions/pokemon-tcg-ai-battle-episodes-2026-06-24']

def ep_path(eid):
    for d in EP_DIRS:
        p = os.path.join(d, f'{eid}.json')
        if os.path.exists(p):
            return p
    return None

d = pd.read_parquet('/kaggle/working/pokemon-20260627-v0-09d2/pokemon-20260627-v0-09d2-decision_rows.parquet')
pairs = d[d['player_archetype'] == ARCH][['episode_id', 'player_index']].drop_duplicates()
pairs = pairs.head(N_EP).values.tolist()
print(f'{ARCH}: {len(pairs)} episode/player pairs sampled')

def cards_of_pokemon(pk):
    out = [pk.get('id')]
    for k in ('energyCards', 'tools', 'preEvolution'):
        for c in (pk.get(k) or []):
            if c and c.get('id'): out.append(c['id'])
    return [x for x in out if x]

ep_freq = Counter()       # in how many episodes a card appears
copy_est = defaultdict(int)  # max simultaneous copies seen
n_ok = 0
for eid, pi in pairs:
    p = ep_path(str(eid))
    if not p:
        continue
    try:
        j = json.load(open(p))
    except Exception:
        continue
    seen_this_ep = set()
    for st in j['steps']:
        if not (isinstance(st, list) and len(st) > pi):
            continue
        ob = (st[pi] or {}).get('observation') or {}
        cur = ob.get('current') or {}
        pls = cur.get('players') or []
        if len(pls) <= pi:
            continue
        me = pls[pi]
        step_cards = Counter()
        for c in (me.get('hand') or []):
            if c and c.get('id'): step_cards[c['id']] += 1
        for c in (me.get('discard') or []):
            if c and c.get('id'): step_cards[c['id']] += 1
        for c in (me.get('prize') or []):
            if c and c.get('id'): step_cards[c['id']] += 1
        for area in ('active', 'bench'):
            for pk in (me.get(area) or []):
                if pk:
                    for cid in cards_of_pokemon(pk):
                        step_cards[cid] += 1
        for cid, cnt in step_cards.items():
            copy_est[cid] = max(copy_est[cid], cnt)
            seen_this_ep.add(cid)
    for cid in seen_this_ep:
        ep_freq[cid] += 1
    n_ok += 1

print(f'parsed {n_ok} episodes, {len(ep_freq)} distinct cards seen')

# core cards: appear in >=25% of episodes
core = [cid for cid, f in ep_freq.items() if f >= max(2, int(0.25 * n_ok))]
core.sort(key=lambda c: -ep_freq[c])

deck = []
ace_used = 0
has_basic_poke = False
for cid in core:
    c = ct.get(cid)
    if c is None:
        continue
    is_basic_energy = (int(c.cardType) == int(CardType.BASIC_ENERGY))
    if c.aceSpec:
        if ace_used >= 1:
            continue
        copies = 1; ace_used += 1
    elif is_basic_energy:
        copies = min(copy_est[cid], 20)
    else:
        copies = min(copy_est[cid], 4)
    copies = max(1, copies)
    if int(c.cardType) == int(CardType.POKEMON) and c.basic:
        has_basic_poke = True
    deck += [cid] * copies

# ensure >=1 basic pokemon
if not has_basic_poke:
    # add the most frequent basic pokemon overall
    for cid, _ in ep_freq.most_common():
        c = ct.get(cid)
        if c and int(c.cardType) == int(CardType.POKEMON) and c.basic:
            deck = [cid] + deck; has_basic_poke = True; break

# trim/pad to 60
if len(deck) > 60:
    deck = deck[:60]
else:
    # pad with the most common basic energy seen, else basic {P} energy (5)
    pad_energy = None
    for cid, _ in ep_freq.most_common():
        c = ct.get(cid)
        if c and int(c.cardType) == int(CardType.BASIC_ENERGY):
            pad_energy = cid; break
    if pad_energy is None:
        pad_energy = 5
    while len(deck) < 60:
        deck.append(pad_energy)

from collections import Counter as C
comp = C(deck)
print('deck size:', len(deck), 'distinct:', len(comp))
print('top composition:', comp.most_common(12))

# validate via engine: Alakazam vs this deck
ALAK = [741,741,741,741,742,742,742,742,743,743,743,305,305,305,66,66,66,140,1231,1231,1231,1231,1225,1225,1225,1225,1182,1182,1182,1184,1184,1086,1086,1086,1086,1152,1152,1152,1152,1079,1079,1079,1081,1081,1081,1081,1129,1097,1156,1174,1266,1266,1266,19,19,19,19,13,5,5]
try:
    obs, sd = game.battle_start(ALAK, deck)
    if obs is None:
        print('ENGINE REJECTED deck: errorPlayer', sd.errorPlayer, 'errorType', sd.errorType)
    else:
        print('ENGINE ACCEPTED deck — battle_start OK')
    game.battle_finish()
except Exception as e:
    print('battle_start exception:', repr(e))

json.dump({'archetype': ARCH, 'deck': deck}, open(f'/kaggle/working/opp_deck_{ARCH}.json', 'w'))
print('saved /kaggle/working/opp_deck_' + ARCH + '.json')
