import os
import sys
from collections import defaultdict

from cg.api import AreaType, CardType, EnergyType, Observation, SelectContext, OptionType, Card, Pokemon, all_card_data, to_observation_class

"""
Alakazam Deck
This deck uses Alakazam's Powerful Hand attack (20 damage per card in hand)
with a draw engine built around Kadabra/Alakazam Psychic Draw, Dudunsparce's
Run Away Draw, and Fezandipiti ex's Flip the Script.
"""

# Hardcoded deck for notebook training/evaluation/submission import safety.
# This avoids FileNotFoundError when confirm/eval imports this module outside
# /kaggle_simulations/agent, where deck.csv does not exist.
my_deck = [741, 741, 741, 741, 742, 742, 742, 742, 743, 743, 743, 305, 305, 305, 66, 66, 66, 140, 1231, 1231, 1231, 1231, 1225, 1225, 1225, 1225, 1182, 1182, 1182, 1184, 1184, 1086, 1086, 1086, 1086, 1152, 1152, 1152, 1152, 1079, 1079, 1079, 1081, 1081, 1081, 1081, 1129, 1097, 1156, 1174, 1266, 1266, 1266, 19, 19, 19, 19, 13, 5, 5]
assert len(my_deck) == 60, "Alakazam rule: hardcoded deck must contain 60 cards"

# Fetch card metadata database and create an ID-to-Card lookup table
all_card = all_card_data()
card_table = {c.cardId: c for c in all_card}

# Decklist
Abra = 741              # x4
Kadabra = 742            # x4
Alakazam = 743           # x3
Dunsparce = 305          # x3
Dudunsparce = 66         # x3
Fezandipiti_ex = 140     # x1
Genesect = 142           # x1
Psyduck = 858            # x1
Shaymin = 343            # x1
Rare_Candy = 1079        # x3
Enhanced_Hammer = 1081   # x4
Buddy_Buddy_Poffin = 1086  # x4
Night_Stretcher = 1097   # x1
Sacred_Ash = 1129        # x1
Poke_Pad = 1152          # x4
Lucky_Helmet = 1156      # x1
Boss_Orders = 1182       # x3
Hilda = 1225             # x4
Dawn = 1231              # x4
Lana_Aid = 1184          # x2
Air_Balloon = 1174       # x1
Nighttime_Mine = 1266    # x3
Battle_Cage = 1264       # not in default deck; retained for compatibility
Basic_Psychic_Energy = 5   # x2
Telepath_Psychic_Energy = 19  # x4
Enriching_Energy = 13    # x1  (ACE SPEC)

# Opponent card IDs to watch for
Duskull = 131
Slowpoke_IDs = (162, 327)
Froakie_IDs = (33, 945)
Wellspring_Mask_Ogerpon_ex = 108
N_Darumaka = 257
Dreepy = 119
Drakloak = 120
Dragapult_ex = 121
Mist_Energy = 11
Rock_Fighting_Energy = 20

# Publicly observed archetype anchors for lightweight opponent-plan suspicion
Hop_Phantump = 878
Hop_Trevenant = 879
Hop_Cramorant = 311
Hop_Snorlax = 304
Postwick = 1255
Riolu = 677
Mega_Lucario_ex = 678
Fighting_Gong = 1142
Basic_Fighting_Energy = 6

# Attack IDs
ATTACK_TELEPORTATION = 1070   # Abra: 10 dmg, cost {P}
ATTACK_SUPER_PSY_BOLT = 1071  # Kadabra: 30 dmg, cost {P}
ATTACK_POWERFUL_HAND = 1072   # Alakazam: 20 per card in hand, cost {P}

# Card ID sets
ABRA_LINE = {Abra, Kadabra, Alakazam}
DUNSPARCE_LINE = {Dunsparce, Dudunsparce}
PSYCHIC_ENERGY_IDS = {Basic_Psychic_Energy, Telepath_Psychic_Energy}

pre_turn = 0
ability_used_dudunsparce = False
ability_used_fezandipiti = False


def get_card(obs: Observation, area: AreaType, index: int, player_index: int) -> Pokemon | Card | None:
    ps = obs.current.players[player_index]
    match area:
        case AreaType.DECK:
            return obs.select.deck[index]
        case AreaType.HAND:
            return ps.hand[index]
        case AreaType.DISCARD:
            return ps.discard[index]
        case AreaType.ACTIVE:
            return ps.active[index]
        case AreaType.BENCH:
            return ps.bench[index]
        case AreaType.PRIZE:
            return ps.prize[index]
        case AreaType.STADIUM:
            return obs.current.stadium[index]
        case AreaType.LOOKING:
            return obs.current.looking[index]
        case _:
            return None


def prize_count(pokemon: Pokemon) -> int:
    data = card_table[pokemon.id]
    count = 3 if data.megaEx else 2 if data.ex else 1
    for card in pokemon.energyCards:
        if card.id == 12:  # Legacy Energy
            count -= 1
    for card in pokemon.tools:
        if card.id == 1172 and "Lillie" in data.name:
            count -= 1
    return max(0, count)


def count_special_defense_energies(pokemon: Pokemon) -> int:
    cnt = 0
    for ec in pokemon.energyCards:
        if ec.id == Mist_Energy or ec.id == Rock_Fighting_Energy:
            cnt += 1
    return cnt


def safe_unique_action(indices: list[int], option_count: int, min_count: int, max_count: int) -> list[int]:
    """Clamp and deduplicate action indices before returning to the game engine."""
    out = []
    seen = set()
    for x in indices:
        try:
            i = int(x)
        except Exception:
            continue
        if 0 <= i < option_count and i not in seen:
            out.append(i)
            seen.add(i)
        if len(out) >= max_count:
            break
    if len(out) < min_count:
        for i in range(option_count):
            if i not in seen:
                out.append(i)
                seen.add(i)
            if len(out) >= min_count:
                break
    return out[:max_count]


def context_threshold(context: SelectContext, safe_draws: int, can_win_this_turn: bool,
                      opponent_archetype: str = 'unknown', has_alakazam: bool = False) -> int:
    """Decision threshold learned from Top Alakazam reports: choose fewer than maxCount.

    Conservative contexts are intentionally harder to select from. This is the rule-level
    translation of the BC/full-rate diagnostics; the model itself is not used at inference.
    """
    if can_win_this_turn:
        return 1
    if context == SelectContext.TO_HAND:
        if safe_draws <= 1:
            return 460
        if safe_draws <= 3:
            return 340
        return 150 if not has_alakazam else 210
    if context == SelectContext.TO_ACTIVE or context == SelectContext.SWITCH:
        return 70
    if context == SelectContext.TO_BENCH:
        return 45 if not has_alakazam else 75
    if context == SelectContext.TO_DECK:
        return 70
    if context == SelectContext.MAIN:
        return 1800
    if context == SelectContext.ATTACH_FROM:
        return 60
    # Numeric/unknown contexts are safer when conservative.
    return 5


def detect_opponent_archetype(op_all_pokemon, stadium_id: int = 0) -> tuple[str, int]:
    """Lightweight public-information archetype suspicion.

    Uses only observed public board cards/stadium. It is intentionally conservative;
    low confidence falls back to generic_control.
    """
    ids = {p.id for p in op_all_pokemon if p is not None}
    scores = defaultdict(int)
    if ids & {Hop_Phantump, Hop_Trevenant, Hop_Cramorant, Hop_Snorlax} or stadium_id == Postwick:
        scores['hop_control'] += 2 + len(ids & {Hop_Phantump, Hop_Trevenant, Hop_Cramorant, Hop_Snorlax})
    if ids & {Abra, Kadabra, Alakazam, Dunsparce, Dudunsparce} or stadium_id == Nighttime_Mine:
        scores['alakazam_mirror'] += 2 + len(ids & {Abra, Kadabra, Alakazam, Dunsparce, Dudunsparce})
    if ids & {Riolu, Mega_Lucario_ex}:
        scores['lucario'] += 2 + len(ids & {Riolu, Mega_Lucario_ex})
    if not scores:
        return 'generic_control', 0
    best, score = max(scores.items(), key=lambda kv: kv[1])
    if score < 2:
        return 'generic_control', score
    return best, int(score)


# v0-05d1 UNKNOWN_0 policy-table assist, distilled from offline PyTorch MLP.
USE_ABSTRACT_OPTION_SIGNATURE = True
USE_UNKNOWN0_POLICY_TABLE = True
UNKNOWN0_POLICY_TABLE = {'END|N1||safe||ko||early': 'END', 'END|N1||safe||ko||mid': 'END', 'END|N1|NO||safe||ko||late': 'END', 'END|N1|NO||safe||ko||mid': 'END', 'END|N1|NO|YES||safe||ko||late': 'END', 'END|N1|NO|YES||safe||ko||mid': 'END', 'END|N1|NO|YES||safe||no_ko||late': 'END', 'END|N1|NO|YES||safe||no_ko||mid': 'END', 'END|N1|YES||safe||ko||early': 'YES', 'END|N1|YES||safe||ko||late': 'END', 'END|N1|YES||safe||ko||mid': 'END', 'END|N3p||safe||ko||late': 'END', 'END|N3p||safe||ko||mid': 'END', 'END|N3p||safe||no_ko||late': 'END', 'END|N3p||safe||no_ko||mid': 'END', 'END|N3p|NO||safe||ko||mid': 'END', 'END|N3p|NO||safe||no_ko||mid': 'END', 'END|N3p|NO|YES||safe||ko||late': 'END', 'END|N3p|NO|YES||safe||ko||mid': 'END', 'END|N3p|NO|YES||safe||no_ko||late': 'END', 'END|N3p|NUMBER||safe||ko||late': 'END', 'END|N3p|NUMBER||safe||ko||mid': 'END', 'END|N3p|YES||safe||ko||mid': 'END', 'END|N3p|YES||safe||no_ko||late': 'END', 'END|N1||*||*||*': 'END', 'END|N1|NO|YES||*||*||*': 'END', 'END|N1|YES||*||*||*': 'END', 'END|N3p||*||*||*': 'END', 'END|N3p|NO||*||*||*': 'END', 'END|N3p|NO|YES||*||*||*': 'END'}
UNKNOWN0_POLICY_STATS = {
    "calls": 0, "unknown0_context": 0, "eligible": 0,
    "key_hit": 0, "signature_fallback_hit": 0, "miss": 0,
    "no_candidate": 0, "selected": 0,
}
UNKNOWN0_POLICY_SELECTED_OPTION_TYPE_COUNTS = {}
UNKNOWN0_POLICY_HIT_KEY_COUNTS = {}


def _unknown0_policy_stat_inc(key, amount=1):
    try:
        UNKNOWN0_POLICY_STATS[key] = UNKNOWN0_POLICY_STATS.get(key, 0) + amount
    except Exception:
        pass


def _unknown0_policy_count_dict(d, key, amount=1):
    try:
        d[key] = d.get(key, 0) + amount
    except Exception:
        pass


def _unknown0_policy_ctx_identifiers(context):
    ids = []
    for attr in ("name", "value"):
        try:
            v = getattr(context, attr, None)
            if v is not None:
                ids.append(str(v))
        except Exception:
            pass
    try:
        s = str(context)
        ids.append(s)
        if "." in s:
            ids.append(s.split(".")[-1])
    except Exception:
        pass
    return set(ids)


def _unknown0_policy_is_context(context):
    ids = _unknown0_policy_ctx_identifiers(context)
    return bool(ids.intersection({"UNKNOWN_0", "UNKNOWN", "0"}))


def _unknown0_policy_option_type_identifiers(option):
    ids = []
    t = getattr(option, "type", None)
    for attr in ("name", "value"):
        try:
            v = getattr(t, attr, None)
            if v is not None:
                ids.append(str(v))
        except Exception:
            pass
    try:
        s = str(t)
        ids.append(s)
        if "." in s:
            ids.append(s.split(".")[-1])
    except Exception:
        pass
    return set(x.upper() if x.isalpha() else x for x in ids if x is not None and str(x) != "")


def _unknown0_policy_signature_forms(select):
    # Build multiple forms so dataset numeric option types like 12|13|14 and runtime enum names can both hit.
    per_option = [_unknown0_policy_option_type_identifiers(o) for o in select.option]
    forms = set()
    for pick in ("name", "value", "str"):
        vals = []
        for o in select.option:
            t = getattr(o, "type", None)
            v = None
            try:
                if pick == "name":
                    v = getattr(t, "name", None)
                elif pick == "value":
                    v = getattr(t, "value", None)
                else:
                    v = str(t).split(".")[-1]
            except Exception:
                v = None
            if v is not None:
                vals.append(str(v).upper() if str(v).isalpha() else str(v))
        if vals:
            forms.add("|".join(sorted(set(vals))))
    # also a broad merged form as last resort
    merged = []
    for ids in per_option:
        if ids:
            # prefer numeric id if present to match observed UNKNOWN_0 training signatures
            nums = sorted([x for x in ids if str(x).isdigit()])
            merged.append(nums[0] if nums else sorted(ids)[0])
    if merged:
        forms.add("|".join(sorted(set(merged))))
    return forms


def _unknown0_policy_abstract_sig(select):
    KEYWORDS = set(["END", "YES", "NO", "NUMBER"])
    per_option = [_unknown0_policy_option_type_identifiers(o) for o in select.option]
    canonical = set()
    for ids in per_option:
        nums = sorted([x for x in ids if x.lstrip("-").isdigit()])
        if nums:
            canonical.add(nums[0])
        else:
            names = sorted([x for x in ids if x.isalpha()])
            if names:
                canonical.add(names[0].upper())
    keywords_found = canonical & KEYWORDS
    n_numeric = len([t for t in canonical if t.lstrip("-").isdigit()])
    n_bucket = "N" + ("0" if n_numeric == 0 else "1" if n_numeric == 1 else "2" if n_numeric == 2 else "3p")
    parts = [n_bucket]
    for kw in ["END", "YES", "NO", "NUMBER"]:
        if kw in keywords_found:
            parts.append(kw)
    return "|".join(sorted(parts))


def _unknown0_policy_select(select, scores, context, turn, deckout_risk_strict, deckout_risk, can_win_this_turn, target_can_kill, desc_indices):
    _unknown0_policy_stat_inc("calls")
    if not USE_UNKNOWN0_POLICY_TABLE or not _unknown0_policy_is_context(context):
        return None
    _unknown0_policy_stat_inc("unknown0_context")
    try:
        if int(select.minCount) != 1 or int(select.maxCount) != 1:
            return None
    except Exception:
        return None
    _unknown0_policy_stat_inc("eligible")
    deck_bucket = "strict" if deckout_risk_strict else "risk" if deckout_risk else "safe"
    ko_bucket = "ko" if (target_can_kill or can_win_this_turn) else "no_ko"
    turn_bucket = "early" if int(turn) <= 2 else "mid" if int(turn) <= 5 else "late"
    preferred = None
    hit_key = None
    signature_fallback = False
    _use_abs = globals().get("USE_ABSTRACT_OPTION_SIGNATURE", False)
    if _use_abs:
        sig = _unknown0_policy_abstract_sig(select)
        for k in [
            f"{sig}||{deck_bucket}||{ko_bucket}||{turn_bucket}",
            f"{sig}||{deck_bucket}||{ko_bucket}||*",
            f"{sig}||{deck_bucket}||*||*",
            f"{sig}||*||*||*",
        ]:
            if k in UNKNOWN0_POLICY_TABLE:
                preferred = UNKNOWN0_POLICY_TABLE[k]
                hit_key = k
                signature_fallback = k.endswith("||*||*||*")
                break
    else:
        sig_forms = _unknown0_policy_signature_forms(select)
        for sig in sorted(sig_forms):
            for k in [
                f"{sig}||{deck_bucket}||{ko_bucket}||{turn_bucket}",
                f"{sig}||{deck_bucket}||{ko_bucket}||*",
                f"{sig}||{deck_bucket}||*||*",
                f"{sig}||*||*||*",
            ]:
                if k in UNKNOWN0_POLICY_TABLE:
                    preferred = UNKNOWN0_POLICY_TABLE[k]
                    hit_key = k
                    signature_fallback = k.endswith("||*||*||*")
                    break
            if preferred is not None:
                break
    if preferred is None:
        _unknown0_policy_stat_inc("miss")
        return None
    _unknown0_policy_stat_inc("key_hit")
    if signature_fallback:
        _unknown0_policy_stat_inc("signature_fallback_hit")
    _unknown0_policy_count_dict(UNKNOWN0_POLICY_HIT_KEY_COUNTS, hit_key)
    candidates = []
    preferred_s = str(preferred).upper() if str(preferred).isalpha() else str(preferred)
    for i, o in enumerate(select.option):
        if preferred_s in _unknown0_policy_option_type_identifiers(o):
            candidates.append(i)
    if not candidates:
        _unknown0_policy_stat_inc("no_candidate")
        return None
    selected_i = max(candidates, key=lambda i: scores[i] if i < len(scores) else -10**18)
    _unknown0_policy_stat_inc("selected")
    _unknown0_policy_count_dict(UNKNOWN0_POLICY_SELECTED_OPTION_TYPE_COUNTS, preferred_s)
    return selected_i

def agent(obs_dict: dict) -> list[int]:
    obs = to_observation_class(obs_dict)
    if obs.select is None:
        return my_deck

    state = obs.current
    select = obs.select
    context = select.context
    my_index = state.yourIndex
    my_state = state.players[my_index]
    op_state = state.players[1 - my_index]
    my_prize_count = len(my_state.prize)

    global pre_turn, ability_used_dudunsparce, ability_used_fezandipiti
    if pre_turn != state.turn:
        pre_turn = state.turn
        ability_used_dudunsparce = False
        ability_used_fezandipiti = False

    # ---- Count cards on field / hand / discard ----
    field_counts = defaultdict(int)
    hand_counts = defaultdict(int)
    discard_counts = defaultdict(int)

    my_field = []  # (field_index, pokemon) where 0=active, 1..=bench
    for card in my_state.active:
        if card is not None:
            field_counts[card.id] += 1
            my_field.append((0, card))
    for idx, card in enumerate(my_state.bench):
        if card is not None:
            field_counts[card.id] += 1
            my_field.append((idx + 1, card))

    for card in my_state.hand:
        hand_counts[card.id] += 1

    for card in my_state.discard:
        discard_counts[card.id] += 1

    abra_line_on_field = field_counts[Abra] + field_counts[Kadabra] + field_counts[Alakazam]
    dunsparce_line_on_field = field_counts[Dunsparce] + field_counts[Dudunsparce]

    # ---- Opponent field analysis ----
    op_all_pokemon = []
    for card in op_state.active:
        if card is not None:
            op_all_pokemon.append(card)
    for card in op_state.bench:
        if card is not None:
            op_all_pokemon.append(card)

    op_has_duskull = any(p.id == Duskull for p in op_all_pokemon)
    op_has_water_threat = any(
        p.id in Slowpoke_IDs or p.id in Froakie_IDs
        or p.id == Wellspring_Mask_Ogerpon_ex or p.id == N_Darumaka
        for p in op_all_pokemon
    )
    op_has_dragapult_line = any(
        p.id in (Dreepy, Drakloak, Dragapult_ex) for p in op_all_pokemon
    )

    # Detect if opponent has used ACE SPEC
    op_used_ace_spec = False
    for log in obs.logs:
        if hasattr(log, 'cardId') and log.cardId is not None:
            cd = card_table.get(log.cardId)
            if cd and cd.aceSpec and hasattr(log, 'playerIndex') and log.playerIndex == (1 - my_index):
                op_used_ace_spec = True

    stadium_id = 0
    for card in state.stadium:
        stadium_id = card.id

    opponent_archetype, opponent_confidence = detect_opponent_archetype(op_all_pokemon, stadium_id)

    bench_count = len(my_state.bench)
    bench_max = my_state.benchMax
    bench_free = bench_max - bench_count

    # ---- Active pokemon info ----
    active_pokemon = my_state.active[0] if my_state.active else None
    active_id = active_pokemon.id if active_pokemon else -1
    active_has_psychic = False
    if active_pokemon:
        for ec in active_pokemon.energyCards:
            if ec.id in PSYCHIC_ENERGY_IDS:
                active_has_psychic = True
                break

    # ---- Opponent active info ----
    op_active = op_state.active[0] if op_state.active else None
    op_active_hp = op_active.hp if op_active else 9999

    # ---- Estimate Powerful Hand damage range ----
    hand_size = len(my_state.hand) if my_state.hand else my_state.handCount

    def estimate_hand_increase():
        """Returns (min_increase, max_increase) of hand size this turn from draw effects."""
        min_inc = 0
        max_inc = 0
        for _, p in my_field:
            if p.id == Abra and hand_counts[Kadabra] > 0:
                max_inc += 1  # evolve Kadabra: hand -1, draw +2 = net +1
            elif p.id == Abra and hand_counts[Rare_Candy] > 0 and hand_counts[Alakazam] > 0:
                max_inc += 1  # Rare Candy + Alakazam: hand -2, draw +3 = net +1
            elif p.id == Kadabra and hand_counts[Alakazam] > 0:
                max_inc += 2  # evolve Alakazam: hand -1, draw +3 = net +2
            elif p.id == Dunsparce and hand_counts[Dudunsparce] > 0:
                max_inc += 1  # evolve: hand -1, ability draw +2 = net +1
            elif p.id == Dudunsparce:
                if not ability_used_dudunsparce:
                    max_inc += 3  # Run Away Draw
            elif p.id == Fezandipiti_ex:
                if not ability_used_fezandipiti:
                    max_inc += 3  # Flip the Script
        if hand_counts[Fezandipiti_ex] > 0 and bench_free > 0 and field_counts[Fezandipiti_ex] == 0:
            max_inc += 2  # play -1, ability +3 = net +2

        # Supporter (only 1 can be used)
        supporter_options = []
        if not state.supporterPlayed:
            if hand_counts[Hilda] > 0:
                supporter_options.append(1)   # play -1, search +2 = net +1
            if hand_counts[Dawn] > 0:
                supporter_options.append(2)   # play -1, search +3 = net +2
            if hand_counts[Boss_Orders] > 0:
                supporter_options.append(-1)  # play -1 = net -1
        if supporter_options:
            max_inc += max(supporter_options)

        # Enriching Energy attach: hand -1, draw +4 = net +3
        if hand_counts[Enriching_Energy] > 0 and not state.energyAttached:
            if active_id == Alakazam and active_has_psychic:
                max_inc += 3
        return min_inc, max_inc

    min_hand_inc, max_hand_inc = estimate_hand_increase()
    max_hand_size = hand_size + max_hand_inc
    min_hand_size = hand_size + min_hand_inc
    max_damage = max_hand_size * 20
    min_damage = min_hand_size * 20

    # ---- Target selection for attack ----
    target_idx = -1       # 0 = active, 1.. = bench
    target_pokemon = None
    target_use_boss = False
    target_can_kill = False
    target_prize_gain = 0
    target_hammer_needed = 0
    use_kadabra_finish = False

    if state.turn >= 2 and op_active is not None:
        # Check Kadabra finisher: opponent active HP <= 30
        if op_active_hp <= 30 and (field_counts[Kadabra] >= 1 or active_id == Kadabra):
            target_idx = 0
            target_pokemon = op_active
            target_use_boss = False
            target_can_kill = True
            target_prize_gain = prize_count(op_active)
            use_kadabra_finish = True
        else:
            # Evaluate all opponent pokemon
            all_op = [(0, op_active)]
            for bi, bp in enumerate(op_state.bench):
                if bp is not None:
                    all_op.append((bi + 1, bp))

            candidates = []
            for oi, pkmn in all_op:
                pz = prize_count(pkmn)
                sp_e = count_special_defense_energies(pkmn)
                eff_max_dmg = max_damage
                hm_need = 0
                if sp_e > 0:
                    if hand_counts[Enhanced_Hammer] >= sp_e:
                        hm_need = sp_e
                        eff_max_dmg = (max_hand_size - hm_need) * 20
                    else:
                        eff_max_dmg = 0
                ck = pkmn.hp <= eff_max_dmg and eff_max_dmg > 0
                candidates.append((oi, pkmn, pz, ck, hm_need))

            # Priority 1: kill wins the game
            win_cands = [(oi, pk, pz, ck, hm) for oi, pk, pz, ck, hm in candidates if ck and my_prize_count <= pz]
            if win_cands:
                # Among winners, prefer active (no boss needed), then highest HP
                best = min(win_cands, key=lambda x: (0 if x[0] == 0 else 1, -x[1].hp))
                target_idx, target_pokemon, target_prize_gain, target_can_kill, target_hammer_needed = best
                target_use_boss = target_idx != 0
            else:
                # Priority 2: killable target with most prizes
                killable = [(oi, pk, pz, ck, hm) for oi, pk, pz, ck, hm in candidates if ck]
                if killable:
                    best = max(killable, key=lambda x: (x[2], x[1].hp))
                    target_idx, target_pokemon, target_prize_gain, target_can_kill, target_hammer_needed = best
                    target_use_boss = target_idx != 0
                else:
                    # Priority 3: just hit active
                    target_idx = 0
                    target_pokemon = op_active
                    target_use_boss = False
                    target_can_kill = False
                    target_prize_gain = 0

    # Should we use Dudunsparce's ability?
    need_dudunsparce_draw = False
    if target_pokemon is not None and target_can_kill:
        needed = target_pokemon.hp
        current_dmg = (hand_size - target_hammer_needed) * 20
        if current_dmg < needed:
            need_dudunsparce_draw = True

    # Do we need to attach energy to the active to retreat?
    need_retreat_energy = False
    if active_pokemon is not None and state.turn >= 2:
        active_is_attacker = (active_id == Alakazam and active_has_psychic) or (use_kadabra_finish and active_id == Kadabra)
        if not active_is_attacker:
            # Check if there's a better attacker on bench
            has_bench_attacker = False
            if use_kadabra_finish and field_counts[Kadabra] >= 1 and active_id != Kadabra:
                has_bench_attacker = True
            elif field_counts[Alakazam] >= 1 and active_id != Alakazam:
                has_bench_attacker = True
            elif field_counts[Kadabra] >= 1 and active_id != Kadabra:
                has_bench_attacker = True
            if has_bench_attacker:
                retreat_cost = card_table[active_pokemon.id].retreatCost
                active_energy_count = len(active_pokemon.energies)
                if active_energy_count < retreat_cost:
                    need_retreat_energy = True

    # Do we need Fezandipiti ex's Flip the Script to kill the target?
    fez_hand_contribution = 0
    if field_counts[Fezandipiti_ex] >= 1 and not ability_used_fezandipiti:
        fez_hand_contribution = 3
    elif hand_counts[Fezandipiti_ex] > 0 and bench_free > 0 and field_counts[Fezandipiti_ex] == 0:
        fez_hand_contribution = 2  # play -1, ability +3 = net +2
    need_fezandipiti_draw = False
    if target_pokemon is not None and target_can_kill and fez_hand_contribution > 0:
        max_damage_without_fez = (max_hand_size - fez_hand_contribution - target_hammer_needed) * 20
        if max_damage_without_fez < target_pokemon.hp:
            need_fezandipiti_draw = True

    # Also allow Fezandipiti if drawing could find key enablers (Boss, Rare Candy, Alakazam, Energy)
    need_fezandipiti_for_setup = False
    if target_pokemon is not None and target_can_kill and fez_hand_contribution > 0 and not need_fezandipiti_draw:
        # Missing Boss's Orders for bench target
        missing_boss = (target_use_boss and hand_counts[Boss_Orders] == 0
                        and not state.supporterPlayed)
        # Check if we have a ready attacker (Alakazam with psychic energy)
        has_ready_attacker = (active_id == Alakazam and active_has_psychic)
        if not has_ready_attacker:
            for _, p in my_field:
                if p.id == Alakazam and any(ec.id in PSYCHIC_ENERGY_IDS for ec in p.energyCards):
                    has_ready_attacker = True
                    break
        missing_attacker = False
        missing_energy = False
        if not has_ready_attacker:
            # Can we set up Alakazam this turn?
            can_evolve_to_alakazam = (field_counts[Kadabra] >= 1 and hand_counts[Alakazam] >= 1)
            can_rare_candy_alakazam = (field_counts[Abra] >= 1 and hand_counts[Rare_Candy] >= 1
                                       and hand_counts[Alakazam] >= 1)
            if not can_evolve_to_alakazam and not can_rare_candy_alakazam:
                # Missing evolution pieces
                if field_counts[Kadabra] >= 1 and hand_counts[Alakazam] == 0:
                    missing_attacker = True
                elif field_counts[Abra] >= 1 and (hand_counts[Rare_Candy] == 0 or hand_counts[Alakazam] == 0):
                    missing_attacker = True
            # Check if energy is available for the attacker
            energy_in_hand = (hand_counts[Basic_Psychic_Energy] + hand_counts[Telepath_Psychic_Energy]
                              + hand_counts[Enriching_Energy])
            if not state.energyAttached and energy_in_hand == 0:
                has_energized = any(
                    p.id in ABRA_LINE and any(ec.id in PSYCHIC_ENERGY_IDS for ec in p.energyCards)
                    for _, p in my_field
                )
                if not has_energized:
                    missing_energy = True
        if missing_boss or missing_attacker or missing_energy:
            need_fezandipiti_for_setup = True

    # Deck safety: don't let deck count drop to <= prize count unless winning this turn.
    can_win_this_turn = target_can_kill and my_prize_count <= target_prize_gain
    deck_count = my_state.deckCount
    # safe_draws: max cards we can draw/search from deck while keeping deck > prize count.
    # Keep one card for the next turn's mandatory draw unless we can win immediately.
    safe_draws = deck_count - my_prize_count - 1 if not can_win_this_turn else 999
    deckout_risk_strict = (safe_draws <= 0 and not can_win_this_turn)
    deckout_risk = (safe_draws <= 3 and not can_win_this_turn)
    setup_incomplete = field_counts[Alakazam] == 0
    need_abra_setup = field_counts[Abra] + field_counts[Kadabra] + field_counts[Alakazam] < 2
    need_dunsparce_setup = field_counts[Dunsparce] + field_counts[Dudunsparce] < 1

    # ---- Score each option ----
    scores = []
    for o in select.option:
        score = 0

        if o.type == OptionType.NUMBER:
            num = int(o.number)
            if deckout_risk and not can_win_this_turn:
                # Prefer smaller optional draw/search numbers when the deck is thin.
                limit = max(0, safe_draws)
                score = 1000 - abs(num - limit) * 150 - max(0, num - limit) * 600
            else:
                score = num

        elif o.type == OptionType.YES:
            score = -1 if deckout_risk_strict else 1

        elif o.type == OptionType.CARD:
            card = get_card(obs, o.area, o.index, o.playerIndex)
            if card is None:
                scores.append(score)
                continue
            energy_count = len(card.energies) if isinstance(card, Pokemon) else 0

            if context == SelectContext.SWITCH or context == SelectContext.TO_ACTIVE:
                if o.playerIndex == my_index:
                    if card.id == Alakazam:
                        score += 100 + energy_count * 10
                    elif card.id == Kadabra:
                        score += 90 if (op_active_hp <= 30) else 30
                    elif card.id == Abra:
                        score += 10
                    elif card.id in DUNSPARCE_LINE:
                        score += 5
                    else:
                        score += 1
                else:
                    if target_use_boss and target_pokemon is not None:
                        if o.index == target_idx - 1:
                            score += 100

            elif context == SelectContext.SETUP_ACTIVE_POKEMON:
                if card.id == Abra:
                    score = 10
                elif card.id == Dunsparce:
                    score = 5
                elif card.id == Psyduck:
                    score = 2
                elif card.id == Shaymin:
                    score = 1

            elif context == SelectContext.SETUP_BENCH_POKEMON:
                if card.id == Abra:
                    cur = field_counts[Abra] + field_counts[Kadabra] + field_counts[Alakazam]
                    score = 200 if cur == 0 else 100 + (3 - cur) * 10
                elif card.id == Dunsparce:
                    score = 150 if dunsparce_line_on_field == 0 else 50

            elif context == SelectContext.TO_HAND:
                score = 200 - hand_counts.get(card.id, 0) * 50
                if card.id == Dudunsparce:
                    score += 80 if (field_counts[Dunsparce] >= 1 and field_counts[Dudunsparce] == 0) else -50
                elif card.id == Kadabra:
                    score += 70 if field_counts[Abra] >= 1 else -20
                elif card.id == Alakazam:
                    score += 60 if (field_counts[Kadabra] >= 1 or field_counts[Abra] >= 1) else -20
                elif card.id == Abra:
                    score += 50 if abra_line_on_field < 3 else -50
                elif card.id == Dunsparce:
                    score += 40 if dunsparce_line_on_field < 2 else -50
                elif card.id in PSYCHIC_ENERGY_IDS:
                    score += 30 if not state.energyAttached else -10
                elif card.id == Enriching_Energy:
                    score += 20
                elif card.id == Rare_Candy:
                    score += 90 if (field_counts[Abra] >= 1 and field_counts[Alakazam] == 0) else -10
                elif card.id == Buddy_Buddy_Poffin:
                    score += 80 if (need_abra_setup or need_dunsparce_setup) else -30
                elif card.id == Hilda:
                    score += 70 if setup_incomplete else -20
                elif card.id == Dawn:
                    score += 80 if setup_incomplete else -20
                elif card.id == Boss_Orders:
                    if target_use_boss and target_can_kill:
                        score += 160
                    elif opponent_archetype in ('alakazam_mirror', 'hop_control') and state.turn >= 4:
                        score += 50
                    else:
                        score -= 40
                elif card.id == Enhanced_Hammer:
                    score += 130 if target_hammer_needed > 0 else (70 if opponent_archetype in ('hop_control', 'lucario') else 20)
                elif card.id == Nighttime_Mine:
                    if stadium_id != Nighttime_Mine:
                        score += 130 if opponent_archetype in ('hop_control', 'lucario', 'alakazam_mirror') else 60
                    else:
                        score -= 80

                if deckout_risk and not can_win_this_turn:
                    key_cards = {Alakazam, Kadabra, Rare_Candy, Boss_Orders, Enhanced_Hammer, Nighttime_Mine, Basic_Psychic_Energy, Telepath_Psychic_Energy}
                    if card.id not in key_cards:
                        score -= 180
                    if deckout_risk_strict and card.id not in key_cards:
                        score = -1

            elif context == SelectContext.ATTACH_FROM:
                if isinstance(card, Pokemon):
                    if need_retreat_energy and o.area == AreaType.ACTIVE:
                        score = 150  # Must attach to active to retreat
                    elif len(card.energyCards) >= 1:
                        score = -1  # Don't attach 2+ energy to the same pokemon
                    elif card.id in ABRA_LINE:
                        score = 100
                        if card.id == Alakazam:
                            score += 20
                        elif card.id == Kadabra:
                            score += 10
                        if o.area == AreaType.ACTIVE:
                            score += 5
                    elif card.id in DUNSPARCE_LINE:
                        score = 50
                    else:
                        score = 10

            elif context == SelectContext.TO_BENCH:
                if card.id == Abra:
                    score = 100
                elif card.id == Dunsparce:
                    score = 80
                elif card.id == Psyduck:
                    if op_has_duskull:
                        score = 60
                    else:
                        score = -1
                elif card.id == Shaymin:
                    if op_has_water_threat:
                        score = 40
                    else:
                        score = -1

            elif context == SelectContext.TO_DECK:
                if card.id in ABRA_LINE:
                    score = 100
                elif card.id in DUNSPARCE_LINE:
                    score = 50
                else:
                    score = 10

        elif o.type == OptionType.PLAY:
            card = get_card(obs, AreaType.HAND, o.index, my_index)
            data = card_table[card.id]

            if data.cardType == CardType.POKEMON:
                score = 20000
                is_early = state.turn <= 2

                if card.id == Abra:
                    if is_early:
                        score += 500
                    elif abra_line_on_field < 3:
                        score += 200
                    elif bench_free <= 1:
                        score = -1
                    else:
                        score += 50

                elif card.id == Dunsparce:
                    if dunsparce_line_on_field < 1:
                        score += 400 if is_early else 100
                    elif dunsparce_line_on_field < 2:
                        score += 50
                    else:
                        score = -1

                elif card.id == Fezandipiti_ex:
                    if need_fezandipiti_draw or need_fezandipiti_for_setup:
                        score += 80 if not is_early else 30
                    else:
                        score = -1  # Don't play unless Flip the Script is needed to kill

                elif card.id == Genesect:
                    if not op_used_ace_spec and (hand_counts[Lucky_Helmet] > 0 or hand_counts[Poke_Pad] > 0):
                        score += 100
                    else:
                        score = -1

                elif card.id == Psyduck:
                    if op_has_duskull:
                        score += 300
                    else:
                        score = -1

                elif card.id == Shaymin:
                    if op_has_water_threat:
                        score += 300
                    else:
                        score = -1

                # Keep at least 1 bench slot free
                if bench_free <= 1 and score > 0:
                    score -= 5000

            else:
                score = 10000

                if card.id == Buddy_Buddy_Poffin:
                    if safe_draws < 2:
                        score = -1  # Deck too thin (searches deck)
                    elif state.turn <= 2:
                        if abra_line_on_field < 3 or dunsparce_line_on_field < 1:
                            score = 18000
                        else:
                            score = 8000
                    else:
                        if abra_line_on_field < 3 or dunsparce_line_on_field < 2:
                            score = 15000
                        elif target_can_kill:
                            score = 8000
                        else:
                            score = -1

                elif card.id == Poke_Pad:
                    if safe_draws < 1:
                        score = -1  # Deck too thin (searches deck)
                    elif state.turn <= 2:
                        score = 17000
                    else:
                        score = 14000 if abra_line_on_field < 3 else 12000

                elif card.id == Rare_Candy:
                    if field_counts[Abra] >= 1 and hand_counts[Alakazam] >= 1 and safe_draws >= 3:
                        score = 16000
                    else:
                        score = -1

                elif card.id == Night_Stretcher:
                    dis_abra = discard_counts[Abra] + discard_counts[Kadabra] + discard_counts[Alakazam]
                    if dis_abra >= 1:
                        score = 13000
                    elif discard_counts[Basic_Psychic_Energy] + discard_counts[Telepath_Psychic_Energy] >= 1:
                        score = 11000
                    else:
                        score = -1

                elif card.id == Sacred_Ash:
                    dis_abra = discard_counts[Abra] + discard_counts[Kadabra] + discard_counts[Alakazam]
                    if dis_abra >= 2:
                        score = 13500
                    elif dis_abra >= 1:
                        score = 11000
                    else:
                        score = -1

                elif card.id == Enhanced_Hammer:
                    if target_hammer_needed > 0:
                        score = 6500
                    else:
                        # Check if any opponent pokemon has special defense energy
                        any_special = any(count_special_defense_energies(p) > 0 for p in op_all_pokemon)
                        if any_special:
                            score = 6200 if opponent_archetype in ('hop_control', 'lucario') else 5000
                        elif opponent_archetype in ('hop_control', 'lucario') and state.turn >= 3:
                            score = 2200
                        else:
                            score = -1

                elif card.id == Lucky_Helmet:
                    score = 7000  # Will be handled via ATTACH

                elif card.id == Boss_Orders:
                    if target_use_boss and target_can_kill:
                        score = 9000 if can_win_this_turn else 6200
                    elif opponent_archetype == 'alakazam_mirror' and state.turn >= 4:
                        score = 2400
                    elif opponent_archetype == 'hop_control' and state.turn >= 4:
                        score = 2200
                    else:
                        score = -1

                elif card.id == Hilda:
                    if safe_draws >= 2:
                        score = 3000
                    else:
                        score = -1

                elif card.id == Dawn:
                    if safe_draws >= 3:
                        score = 3100
                    else:
                        score = -1

                elif card.id == Lana_Aid:
                    dis_line = discard_counts[Abra] + discard_counts[Kadabra] + discard_counts[Alakazam] + discard_counts[Dunsparce] + discard_counts[Dudunsparce]
                    dis_energy = discard_counts[Basic_Psychic_Energy] + discard_counts[Telepath_Psychic_Energy]
                    if dis_line + dis_energy >= 2:
                        score = 7600
                    elif state.turn >= 5 and dis_line + dis_energy >= 1:
                        score = 4200
                    else:
                        score = -1

                elif card.id == Nighttime_Mine:
                    # Prefer Nighttime Mine in the control mirrors / Tera-energy tax matchups.
                    if stadium_id != Nighttime_Mine:
                        if opponent_archetype in ('hop_control', 'lucario'):
                            score = 15500
                        elif opponent_archetype == 'alakazam_mirror' and state.turn >= 3:
                            score = 9800
                        elif op_has_dragapult_line or target_can_kill:
                            score = 15000
                        elif stadium_id != 0:
                            score = 7600
                        elif state.turn >= 3:
                            score = 5300
                        else:
                            score = -1
                    else:
                        score = -1

                elif card.id == Battle_Cage:
                    if op_has_dragapult_line:
                        score = 9000
                    elif stadium_id != 0:
                        score = 4000
                    else:
                        score = -1

        elif o.type == OptionType.ATTACH:
            card = get_card(obs, AreaType.HAND, o.index, my_index)
            pokemon = get_card(obs, o.inPlayArea, o.inPlayIndex, my_index)

            if card.id == Air_Balloon:
                if o.inPlayArea == AreaType.ACTIVE and active_id not in (Alakazam, Kadabra):
                    score = 8800
                elif pokemon.id in ABRA_LINE and len(getattr(pokemon, 'tools', [])) == 0:
                    score = 5800
                else:
                    score = -1

            elif card.id == Lucky_Helmet:
                score = 7000
                if pokemon.id == Genesect and not op_used_ace_spec:
                    score += 300
                elif o.inPlayArea == AreaType.ACTIVE:
                    score += 200
                else:
                    score += 50

            elif card.id in PSYCHIC_ENERGY_IDS:
                if need_retreat_energy and o.inPlayArea == AreaType.ACTIVE:
                    score = 9500  # Must attach to active to retreat
                elif len(pokemon.energyCards) >= 1:
                    score = -1  # Don't attach 2+ energy to the same pokemon
                elif pokemon.id in ABRA_LINE:
                    score = 8000
                    if pokemon.id == Alakazam:
                        score += 30
                    elif pokemon.id == Kadabra:
                        score += 20
                    elif pokemon.id == Abra:
                        score += 10
                    if o.inPlayArea == AreaType.ACTIVE:
                        score += 5
                else:
                    score = -1
                # Telepath Psychic Energy searches 2 from deck
                if card.id == Telepath_Psychic_Energy and safe_draws < 2 and score > 0:
                    score = -1

            elif card.id == Enriching_Energy:
                if need_retreat_energy and o.inPlayArea == AreaType.ACTIVE:
                    score = 9500  # Must attach to active to retreat
                elif len(pokemon.energyCards) >= 1:
                    score = -1  # Don't attach 2+ energy to the same pokemon
                elif pokemon.id in DUNSPARCE_LINE:
                    score = 8500
                    if pokemon.id == Dudunsparce:
                        score += 10
                else:
                    score = -1
                # Enriching Energy draws 4 from deck
                if card.id == Enriching_Energy and safe_draws < 4 and score > 0:
                    score = -1

        elif o.type == OptionType.EVOLVE:
            card = get_card(obs, AreaType.HAND, o.index, my_index)
            pokemon = get_card(obs, o.inPlayArea, o.inPlayIndex, my_index)
            score = 9000

            if card.id == Alakazam:
                if safe_draws < 3:
                    score = -1  # Deck too thin for Psychic Draw (3 cards)
                elif o.inPlayArea == AreaType.ACTIVE:
                    score += 200  # Active Alakazam = highest
                else:
                    score += 50  # Bench Alakazam
                score += len(pokemon.energies) * 10

            elif card.id == Kadabra:
                if safe_draws < 2:
                    score = -1  # Deck too thin for Psychic Draw (2 cards)
                else:
                    score += 100
                    if len(pokemon.energies) == 0:
                        score += 50  # Evolve non-energy Abra first
                    else:
                        score -= 20
                        if hand_counts[Rare_Candy] > 0 and hand_counts[Alakazam] > 0:
                            score -= 100  # Save energy Abra for Rare Candy -> Alakazam

            elif card.id == Dudunsparce:
                if safe_draws < 2:
                    score = -1  # Deck too thin for draw on evolve
                else:
                    score += 80

        elif o.type == OptionType.ABILITY:
            card = get_card(obs, o.area, o.index, my_index)
            if card is None:
                scores.append(score)
                continue

            if card.id == Dudunsparce:
                if need_dudunsparce_draw:
                    if safe_draws >= 3:
                        score = 30000
                    else:
                        score = -1  # Deck too thin
                else:
                    score = -1
            elif card.id == Fezandipiti_ex:
                if (need_fezandipiti_draw or need_fezandipiti_for_setup) and safe_draws >= 3:
                    score = 29000
                else:
                    score = -1  # Don't use unless needed to kill target
            elif card.id == Battle_Cage:
                score = 1
            else:
                score = 28000
                if deckout_risk_strict and not can_win_this_turn:
                    score = -1
                elif deckout_risk and not can_win_this_turn:
                    score -= 12000

        elif o.type == OptionType.RETREAT:
            if active_id == Alakazam and active_has_psychic:
                score = -1
            elif use_kadabra_finish and active_id != Kadabra and field_counts[Kadabra] >= 1:
                score = 2500  # Retreat to bring Kadabra forward for finish
            elif active_id in (Abra, Dunsparce, Dudunsparce, Psyduck, Shaymin, Genesect):
                if field_counts[Alakazam] >= 1 or field_counts[Kadabra] >= 1:
                    score = 2000
                else:
                    score = -1
            else:
                score = -1

        elif o.type == OptionType.ATTACK:
            score = 1000
            if o.attackId == ATTACK_POWERFUL_HAND:
                score += 500
                if target_can_kill:
                    score += 5000 if can_win_this_turn else 2600
                elif active_id == Alakazam and active_has_psychic:
                    score += 300
            elif o.attackId == ATTACK_SUPER_PSY_BOLT:
                if op_active_hp <= 30:
                    score += 600  # Kadabra finisher
                else:
                    score += 100
            elif o.attackId == ATTACK_TELEPORTATION:
                score += 50

        scores.append(score)

    # Select in descending order of score.
    # Do not blindly return maxCount: Top control agents often choose fewer than allowed.
    desc_indices = [i for i, _ in sorted(enumerate(scores), key=lambda x: x[1], reverse=True)]
    min_count = max(0, int(select.minCount))
    max_count = min(len(select.option), max(0, int(select.maxCount)))
    if max_count <= 0:
        return []
    unknown0_policy_selected = _unknown0_policy_select(
        select, scores, context, state.turn, deckout_risk_strict, deckout_risk,
        can_win_this_turn, target_can_kill, desc_indices
    ) if 'USE_UNKNOWN0_POLICY_TABLE' in globals() else None
    if unknown0_policy_selected is not None:
        selected = safe_unique_action([unknown0_policy_selected], len(select.option), min_count, max_count)
        if len(selected) >= min_count:
            return selected
    threshold = context_threshold(context, safe_draws, can_win_this_turn, opponent_archetype, field_counts[Alakazam] > 0)
    eligible = [i for i in desc_indices if scores[i] >= threshold]
    # For optional contexts, allow no-op when all choices are weak.
    selected = eligible[:max_count]
    if len(selected) < min_count:
        selected = desc_indices[:min_count]
    selected = safe_unique_action(selected, len(select.option), min_count, max_count)

    if context == SelectContext.MAIN and selected:
        o = select.option[selected[0]]
        if o.type == OptionType.ABILITY:
            card = get_card(obs, o.area, o.index, my_index)
            if card is not None:
                if card.id == Dudunsparce:
                    ability_used_dudunsparce = True
                elif card.id == Fezandipiti_ex:
                    ability_used_fezandipiti = True

    return selected
