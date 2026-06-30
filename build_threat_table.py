"""v13d4: build archetype -> typical-threat table from 28th decklists.
Per deck: the deck's biggest punch = max attack damage across its Pokemon + that attack's energy cost.
Per archetype: prevalence-weighted MEDIAN of (max_dmg, cost) = the typical threat a random opponent
of that archetype presents. This is the 'deck-pool distribution' feature for early opponent prediction."""
import ast, json, sys
sys.path.insert(0, '/kaggle/input/competitions/pokemon-tcg-ai-battle/sample_submission')
import pandas as pd
from cg.api import all_card_data, all_attack

CT = {c.cardId: c for c in all_card_data()}
ATK = {a.attackId: (int(getattr(a, 'damage', 0) or 0), len(getattr(a, 'energies', None) or [])) for a in all_attack()}

def deck_threat(cards):
    """Biggest punch in this deck: (max_dmg, cost_of_that_attack)."""
    best = (0, 99)
    for cid in set(cards):
        cd = CT.get(cid)
        if cd is None: continue
        for aid in (getattr(cd, 'attacks', None) or []):
            dmg, cost = ATK.get(aid, (0, 99))
            if dmg > best[0] or (dmg == best[0] and cost < best[1]):
                best = (dmg, cost)
    return best

df = pd.read_parquet('/kaggle/working/pokemon-20260628-v0-base28/pokemon-20260628-v0-base28-decklists.parquet')
df = df[df['deck_valid'] == True]
ARCH_MAP = {'Alakazam': 'alakazam_mirror', 'Mega_Lucario': 'lucario', 'Hop_Trevenant': 'hop_control', 'Other': 'generic_control'}

# per-deck threat, weighted by occurrence
from collections import defaultdict
buckets = defaultdict(list)  # label -> list of (dmg, cost) per deck-occurrence
all_threats = []
for _, row in df.iterrows():
    try:
        cards = ast.literal_eval(row['deck_list'])
    except Exception:
        continue
    dmg, cost = deck_threat(cards)
    if dmg <= 0: continue
    label = ARCH_MAP.get(row['archetype'], 'generic_control')
    buckets[label].append((dmg, cost))
    all_threats.append((dmg, cost))

def med(xs):
    xs = sorted(xs); n = len(xs)
    return xs[n // 2] if n else 0

table = {}
for label, vals in buckets.items():
    dmgs = [d for d, c in vals]; costs = [c for d, c in vals]
    table[label] = [med(dmgs), med(costs), len(vals)]
# field default
table['_field'] = [med([d for d, c in all_threats]), med([c for d, c in all_threats]), len(all_threats)]
# also report distribution of cost (clock proxy) per label
print('archetype -> [median_max_dmg, median_cost, n_decks]')
for k, v in sorted(table.items(), key=lambda kv: -kv[1][2]):
    print(f'  {k:18}: dmg={v[0]:4} cost={v[1]} n={v[2]}')
json.dump({k: v[:2] for k, v in table.items()}, open('/kaggle/working/archetype_threat.json', 'w'))
print('written /kaggle/working/archetype_threat.json')
