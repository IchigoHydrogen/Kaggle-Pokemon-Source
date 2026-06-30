"""Generic deck-agnostic SENSIBLE rule-base policy (rollout policy / pilot).
Replaces generic_fwdsim's weak greedy rollout. Plays any deck reasonably:
set up (evolve / attach energy to the best attacker / develop) then attack
(lethal preferred), with sensible non-ctx0 handling. Used as a STRONG opponent
pilot for the league. Aggressive decks (Lucario) should be near-optimal."""
import sys
sys.path.insert(0, '/kaggle/input/competitions/pokemon-tcg-ai-battle/sample_submission')
from cg.api import (AreaType, CardType, SelectContext, OptionType, Observation,
                    Card, Pokemon, all_card_data, all_attack, to_observation_class)
CT = {c.cardId: c for c in all_card_data()}
ATK = {a.attackId: (int(getattr(a, 'damage', 0) or 0), len(getattr(a, 'energies', None) or []))
       for a in all_attack()}

def get_card(obs, area, index, pidx):
    st = obs.current; ps = st.players[pidx]
    try:
        if area == AreaType.HAND: return ps.hand[index]
        if area == AreaType.ACTIVE: return ps.active[index]
        if area == AreaType.BENCH: return ps.bench[index]
        if area == AreaType.DISCARD: return ps.discard[index]
        if area == AreaType.DECK: return ps.deck[index]
    except Exception:
        return None
    return None

def pmax_dmg(cid):
    cd = CT.get(cid)
    if cd is None: return 0
    ds = [ATK.get(a, (0, 99))[0] for a in (getattr(cd, 'attacks', None) or [])]
    return max(ds) if ds else 0

def is_pokemon(cid):
    cd = CT.get(cid)
    return cd is not None and getattr(cd, 'cardType', None) == CardType.POKEMON

def is_energy(cid):
    cd = CT.get(cid)
    return cd is not None and getattr(cd, 'cardType', None) == CardType.ENERGY

def sensible_agent(obs_dict):
    obs = to_observation_class(obs_dict)
    sel = obs.select
    if sel is None: return []
    st = obs.current; ctx = sel.context; p = st.yourIndex
    me = st.players[p]; op = st.players[1 - p]
    opa = op.active[0] if op.active else None
    op_hp = int(getattr(opa, 'hp', 0) or 0) if opa else 0
    opts = sel.option
    scores = []
    for o in opts:
        s = 50.0  # default
        t = getattr(o, 'type', None)
        try:
            if t == OptionType.ATTACK:
                dmg, cost = ATK.get(getattr(o, 'attackId', -1), (0, 99))
                s = 1500.0 + dmg                      # attack (after setup)
                if op_hp and dmg >= op_hp: s = 90000.0 + dmg   # LETHAL: top priority
            elif t == OptionType.EVOLVE:
                s = 6000.0                            # evolve toward stronger attacker
            elif t == OptionType.ATTACH:
                tgt = get_card(obs, getattr(o, 'area', None), getattr(o, 'index', 0), p)
                s = 4000.0 + (pmax_dmg(tgt.id) if tgt is not None and hasattr(tgt, 'id') else 0)
            elif t == OptionType.ABILITY:
                s = 3000.0                            # use abilities (draw/accel)
            elif t == OptionType.PLAY:
                card = get_card(obs, AreaType.HAND, getattr(o, 'index', 0), p)
                cid = getattr(card, 'id', None) if card is not None else None
                if cid is not None and is_pokemon(cid):
                    s = 2500.0 + pmax_dmg(cid) * 0.1  # develop, prefer attackers
                else:
                    s = 2000.0                        # trainer (draw/search)
            elif t == OptionType.RETREAT:
                s = 10.0
            else:
                # non-ctx0 resolution contexts: prefer useful targets
                tgt = get_card(obs, getattr(o, 'area', None), getattr(o, 'index', 0), p)
                cid = getattr(tgt, 'id', None) if tgt is not None else None
                if ctx == SelectContext.TO_ACTIVE:   # promote: best attacker forward
                    s = 100.0 + pmax_dmg(cid) if cid else 100.0
                elif ctx in (SelectContext.TO_HAND, SelectContext.TO_BENCH):
                    s = 100.0 + (pmax_dmg(cid) if cid and is_pokemon(cid) else (30.0 if cid and is_energy(cid) else 10.0))
                else:
                    s = 50.0
        except Exception:
            s = 50.0
        scores.append(s)
    desc = sorted(range(len(opts)), key=lambda i: scores[i], reverse=True)
    mn = max(0, int(getattr(sel, 'minCount', 1) or 0))
    mx = min(len(opts), max(0, int(getattr(sel, 'maxCount', 1) or 1)))
    if mx <= 0: return []
    sel_idx = desc[:max(mn, 1)] if mx >= 1 else []
    return sel_idx[:max(1, mx)] if mx >= 1 else sel_idx[:mn]

if __name__ == '__main__':
    print('sensible_agent loaded; cards', len(CT), 'attacks', len(ATK))
    print('Mega Lucario pmax_dmg(678)=', pmax_dmg(678), '(should be 270 Mega Brave)')
    print('Riolu pmax_dmg(677)=', pmax_dmg(677))
