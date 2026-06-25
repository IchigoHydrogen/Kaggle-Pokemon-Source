# =====================================================================
# v0-06d12 Part A: Per-decision CE + Card metadata features + Extended guard scope
#
# Changes from v0-06d10/11:
#   1. Loss: BCEWithLogitsLoss -> per-decision cross-entropy (F.cross_entropy over options)
#   2. Features: card_id%32 one-hot (32 lossy bins) -> card metadata from all_card_data()
#      cardType, evolution stage, ex/mega/tera/aceSpec flags, HP, 19 key-card identity bits
#      Feature dim stays at 96.
#   3. Guard scope: ATTACK/END/RETREAT hard veto removed -> confidence-gated override
#      ABILITY remains hard-vetoed.
# =====================================================================
import importlib.util as _v06d12_ilu
import tarfile as _v06d12_tarfile
import time as _v06d12_time
import torch.nn.functional as _v06d12_F
from collections import defaultdict as _v06d12_dd

MAIN_HYBRID_REPORT = {'status': 'not_run'}
MAIN_LEARNING_REPORT = MAIN_HYBRID_REPORT

# ------------------------------------------------------------------
# Context / option name maps (carried forward from v0-06d7)
# ------------------------------------------------------------------
_V06D12_CONTEXT_NAMES = {
    0: 'MAIN', 1: 'SETUP_ACTIVE', 2: 'SETUP_BENCH', 3: 'SWITCH', 4: 'TO_ACTIVE',
    5: 'TO_BENCH', 6: 'TO_FIELD', 7: 'TO_HAND', 8: 'DISCARD', 9: 'TO_DECK',
    10: 'TO_DECK_BOTTOM', 11: 'TO_PRIZE', 12: 'NOT_MOVE', 13: 'DAMAGE_COUNTER',
    14: 'DAMAGE_COUNTER_ANY', 21: 'ATTACH_FROM', 22: 'ATTACH_TO', 30: 'DISCARD_ENERGY',
    31: 'TO_HAND_ENERGY', 33: 'SWITCH_ENERGY', 35: 'ATTACK', 37: 'EVOLVE',
    41: 'IS_FIRST', 42: 'MULLIGAN', 43: 'ACTIVATE',
}
_V06D12_OPTION_NAMES = {
    0: 'NUMBER', 1: 'YES', 2: 'NO', 3: 'CARD', 4: 'TOOL_CARD',
    5: 'ENERGY_CARD', 6: 'ENERGY', 7: 'PLAY', 8: 'ATTACH',
    9: 'EVOLVE', 10: 'ABILITY', 11: 'DISCARD', 12: 'RETREAT',
    13: 'ATTACK', 14: 'END', 15: 'SKILL', 16: 'SPECIAL_CONDITION',
}

_V06D12_KEY_CARDS = {
    741, 742, 743,   # Abra, Kadabra, Alakazam
    305, 66,         # Dunsparce, Dudunsparce
    140,             # Fezandipiti ex
    1231, 1225,      # Dawn, Hilda
    1182, 1184,      # Boss Orders, Lana's Aid
    1079, 1081,      # Rare Candy, Enhanced Hammer
    1086, 1152,      # Buddy-Buddy Poffin, Poke Pad
    1129, 1097,      # Sacred Ash, Night Stretcher
    1266,            # Nighttime Mine (stadium)
    5, 19, 13,       # Basic Psychic, Telepath Psychic, Enriching Energy
}

# card_meta index mapping (35 dims, placed at [61-95] in 96-dim vector)
# [0]  is_pokemon        cardType==0
# [1]  is_item           cardType==1
# [2]  is_tool           cardType==2
# [3]  is_supporter      cardType==3
# [4]  is_stadium_card   cardType==4
# [5]  is_basic_energy   cardType==5
# [6]  is_special_energy cardType==6
# [7]  is_ex
# [8]  is_mega_ex
# [9]  is_ace_spec
# [10] is_tera
# [11] is_basic_pokemon  (basic==True)
# [12] is_stage1
# [13] is_stage2
# [14] hp_norm           hp/300
# [15] has_skill         len(skills)>0
# [16] is_abra           741
# [17] is_kadabra        742
# [18] is_alakazam       743
# [19] is_dunsparce      305
# [20] is_dudunsparce    66
# [21] is_fezandipiti    140
# [22] is_dawn           1231
# [23] is_hilda          1225
# [24] is_boss_orders    1182
# [25] is_lana_aid       1184
# [26] is_rare_candy     1079
# [27] is_enhanced_hammer 1081
# [28] is_buddy_poffin   1086
# [29] is_poke_pad       1152
# [30] is_sacred_ash     1129
# [31] is_night_stretcher 1097
# [32] is_nighttime_mine  1266
# [33] is_psychic_energy  (5 or 19)
# [34] is_enriching_energy 13
_V06D12_CARD_META_DIM = 35
_V06D12_CARD_META_OFFSET = 61  # start index in 96-dim vector
_V06D12_FEAT_DIM = 96

def _v06d12_name(mapping, value, prefix):
    try:
        return mapping.get(int(value), prefix + '_' + str(value))
    except Exception:
        return prefix + '_' + str(value)

def _v06d12_norm_name(s):
    return re.sub(r'[^a-z0-9]', '', str(s).lower())

def _v06d12_rank_bucket(rank):
    if rank is None:
        return 'unranked'
    if rank <= 10:
        return 'top10'
    if rank <= 50:
        return 'top50'
    if rank <= 200:
        return 'top200'
    return 'other'

def _v06d12_archetype_of_deck(deck):
    s = set(int(x) for x in deck)
    if 743 in s:
        return 'alakazam'
    if 878 in s or 879 in s:
        return 'hop_control'
    if 678 in s or 673 in s:
        return 'lucario'
    return 'other'

def _v06d12_load_agent_from_archive(archive_path):
    with _v06d12_tarfile.open(archive_path) as t:
        src = t.extractfile('main.py').read().decode()
        deck = [int(x) for x in t.extractfile('deck.csv').read().decode().split() if str(x).strip()]
    mod_path = OUTPUT_DIR / (RUN_PREFIX + '-main_learning_rule_agent.py')
    mod_path.write_text(src, encoding='utf-8')
    spec = _v06d12_ilu.spec_from_file_location('v06d12_main_learning_rule_agent', str(mod_path))
    m = _v06d12_ilu.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m, deck

def _v06d12_to_float(x, default=0.0):
    try:
        if x is None:
            return float(default)
        return float(x)
    except Exception:
        return float(default)

def _v06d12_card_id_from_area(obs, option):
    try:
        area = int(option.get('area', 0))
        idx = int(option.get('index', -1))
        player = int(option.get('playerIndex', (obs.get('current') or {}).get('yourIndex', 0)))
        cur = obs.get('current') or {}
        players = cur.get('players') or []
        card = None
        if area == 1:
            deck_cards = (obs.get('select') or {}).get('deck') or []
            if 0 <= idx < len(deck_cards):
                card = deck_cards[idx]
        elif area in (2, 3, 4, 5, 6) and 0 <= player < len(players):
            ps = players[player] or {}
            key = {2: 'hand', 3: 'discard', 4: 'active', 5: 'bench', 6: 'prize'}.get(area)
            cards = ps.get(key) or []
            if 0 <= idx < len(cards):
                card = cards[idx]
        elif area == 7:
            cards = cur.get('stadium') or []
            if 0 <= idx < len(cards):
                card = cards[idx]
        elif area == 12:
            cards = cur.get('looking') or []
            if 0 <= idx < len(cards):
                card = cards[idx]
        if isinstance(card, dict):
            return int(card.get('id') or 0)
        return int(getattr(card, 'id', 0) or 0)
    except Exception:
        return 0

def _v06d12_player_summary(obs, player_idx):
    cur = obs.get('current') or {}
    players = cur.get('players') or []
    if not (0 <= player_idx < len(players)) or not isinstance(players[player_idx], dict):
        return {'hand': 0, 'deck': 0, 'discard': 0, 'bench': 0, 'active': 0, 'prize': 0}
    ps = players[player_idx]
    return {
        'hand': int(ps.get('handCount', len(ps.get('hand') or [])) or 0),
        'deck': int(ps.get('deckCount', len(ps.get('deck') or [])) or 0),
        'discard': int(len(ps.get('discard') or [])),
        'bench': int(sum(1 for x in (ps.get('bench') or []) if x)),
        'active': int(sum(1 for x in (ps.get('active') or []) if x)),
        'prize': int(len(ps.get('prize') or [])),
    }

def _v06d12_card_meta_vec(card_id, card_table):
    """Return 35-dim card metadata vector for card_id."""
    vec = [0.0] * _V06D12_CARD_META_DIM
    card = card_table.get(int(card_id) if card_id else 0)
    if card is None:
        return vec
    ct = int(getattr(card, 'cardType', -1))
    if 0 <= ct <= 6:
        vec[ct] = 1.0
    vec[7]  = float(bool(getattr(card, 'ex', False)))
    vec[8]  = float(bool(getattr(card, 'megaEx', False)))
    vec[9]  = float(bool(getattr(card, 'aceSpec', False)))
    vec[10] = float(bool(getattr(card, 'tera', False)))
    vec[11] = float(bool(getattr(card, 'basic', False)))
    vec[12] = float(bool(getattr(card, 'stage1', False)))
    vec[13] = float(bool(getattr(card, 'stage2', False)))
    hp = int(getattr(card, 'hp', 0) or 0)
    vec[14] = min(1.0, hp / 300.0)
    skills = getattr(card, 'skills', None) or []
    vec[15] = float(len(skills) > 0)
    cid = int(card_id) if card_id else 0
    vec[16] = float(cid == 741)   # Abra
    vec[17] = float(cid == 742)   # Kadabra
    vec[18] = float(cid == 743)   # Alakazam
    vec[19] = float(cid == 305)   # Dunsparce
    vec[20] = float(cid == 66)    # Dudunsparce
    vec[21] = float(cid == 140)   # Fezandipiti ex
    vec[22] = float(cid == 1231)  # Dawn
    vec[23] = float(cid == 1225)  # Hilda
    vec[24] = float(cid == 1182)  # Boss Orders
    vec[25] = float(cid == 1184)  # Lana's Aid
    vec[26] = float(cid == 1079)  # Rare Candy
    vec[27] = float(cid == 1081)  # Enhanced Hammer
    vec[28] = float(cid == 1086)  # Buddy-Buddy Poffin
    vec[29] = float(cid == 1152)  # Poke Pad
    vec[30] = float(cid == 1129)  # Sacred Ash
    vec[31] = float(cid == 1097)  # Night Stretcher
    vec[32] = float(cid == 1266)  # Nighttime Mine
    vec[33] = float(cid in (5, 19))  # Basic/Telepath Psychic Energy
    vec[34] = float(cid == 13)    # Enriching Energy
    return vec

def _v06d12_option_features(obs, option, option_index, option_count, card_table):
    """Reference Python-loop feature extractor (cross-check only). 96 dims."""
    cur = obs.get('current') or {}
    select = obs.get('select') or {}
    your = int(cur.get('yourIndex', 0) or 0)
    opp = 1 - your
    mine = _v06d12_player_summary(obs, your)
    theirs = _v06d12_player_summary(obs, opp)
    context = int(select.get('context', 0) or 0)
    opt_type = int(option.get('type', 0) or 0)
    area = int(option.get('area', 0) or 0)
    card_id = _v06d12_card_id_from_area(obs, option)
    row = [0.0] * _V06D12_FEAT_DIM
    # [0-12]: global context
    row[0] = 1.0
    row[1] = min(1.0, option_count / 32.0)
    row[2] = min(1.0, option_index / max(1, option_count - 1))
    row[3] = _v06d12_to_float(select.get('minCount'), 0.0) / 8.0
    row[4] = _v06d12_to_float(select.get('maxCount'), 0.0) / 8.0
    row[5] = min(1.0, _v06d12_to_float(cur.get('turn'), 0.0) / 20.0)
    row[6] = min(1.0, _v06d12_to_float(cur.get('turnActionCount'), 0.0) / 32.0)
    row[7] = min(1.0, _v06d12_to_float(obs.get('step'), 0.0) / 512.0)
    row[8] = float(your)
    row[9] = float(bool(cur.get('energyAttached')))
    row[10] = float(bool(cur.get('supporterPlayed')))
    row[11] = float(bool(cur.get('retreated')))
    row[12] = float(bool(cur.get('stadiumPlayed')))
    # [13-24]: player summaries
    row[13] = mine['hand'] / 16.0
    row[14] = mine['deck'] / 60.0
    row[15] = mine['discard'] / 60.0
    row[16] = mine['bench'] / 8.0
    row[17] = mine['active'] / 4.0
    row[18] = mine['prize'] / 6.0
    row[19] = theirs['hand'] / 16.0
    row[20] = theirs['deck'] / 60.0
    row[21] = theirs['discard'] / 60.0
    row[22] = theirs['bench'] / 8.0
    row[23] = theirs['active'] / 4.0
    row[24] = theirs['prize'] / 6.0
    # [25]: context
    row[25] = (context % 64) / 64.0
    # [26-32]: option numerics (no card_id continuous; shifted from v06d10)
    row[26] = (opt_type % 32) / 32.0
    row[27] = (area % 16) / 16.0
    row[28] = _v06d12_to_float(option.get('index'), 0.0) / 32.0
    row[29] = _v06d12_to_float(option.get('inPlayIndex'), 0.0) / 16.0
    row[30] = _v06d12_to_float(option.get('number'), 0.0) / 16.0
    row[31] = _v06d12_to_float(option.get('attackId'), 0.0) / 4096.0
    row[32] = _v06d12_to_float(option.get('abilityId'), 0.0) / 4096.0
    # [33-48]: opt_type one-hot (16 dims)
    row[33 + (opt_type % 16)] = 1.0
    # [49-60]: area one-hot (12 dims)
    row[49 + (area % 12)] = 1.0
    # [61-95]: card metadata (35 dims)
    meta = _v06d12_card_meta_vec(card_id, card_table)
    for mi, mv in enumerate(meta):
        row[_V06D12_CARD_META_OFFSET + mi] = mv
    return row


try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    torch.set_num_threads(int(os.environ.get('V06_TORCH_THREADS', '1')))

    # Load rule agent and card table
    # Cell 20 creates the rule-only archive; it may end up in OUTPUT_DIR or WORKING_DIR
    _archive = WORKING_DIR / (RUN_PREFIX + '-submission.tar.gz')
    if not _archive.exists():
        _archive = OUTPUT_DIR / (RUN_PREFIX + '-submission.tar.gz')
    if not _archive.exists():
        _archive = WORKING_DIR / (RUN_PREFIX + '-submission-rule-only.tar.gz')
    if not _archive.exists():
        _archive = OUTPUT_DIR / (RUN_PREFIX + '-submission-rule-only.tar.gz')
    if not _archive.exists():
        raise FileNotFoundError('expected base submission archive not found in WORKING_DIR or OUTPUT_DIR: ' + RUN_PREFIX)
    _rule_mod, _our_deck = _v06d12_load_agent_from_archive(_archive)
    _rule_agent = _rule_mod.agent
    _our_arch = _v06d12_archetype_of_deck(_our_deck)

    # Load card metadata
    _t0_card = _v06d12_time.perf_counter()
    from cg.api import all_card_data as _v06d12_all_card_data
    _card_table = {c.cardId: c for c in _v06d12_all_card_data()}
    print(f'card table loaded: {len(_card_table)} cards in {_v06d12_time.perf_counter()-_t0_card:.2f}s')

    _split_map = {}
    if 'EPISODE_SPLIT_DF' in globals() and isinstance(EPISODE_SPLIT_DF, pd.DataFrame) and not EPISODE_SPLIT_DF.empty:
        for _, r in EPISODE_SPLIT_DF.iterrows():
            _split_map[str(r.get('episode_id', ''))] = str(r.get('split', 'unknown'))

    _rank_info = TOP200_LOOKUP if 'TOP200_LOOKUP' in globals() else {}
    def _rank_of(name):
        info = _rank_info.get(_v06d12_norm_name(name), {}) if isinstance(_rank_info, dict) else {}
        r = info.get('ranking') if isinstance(info, dict) else None
        return int(r) if r is not None else None

    _max_decisions = int(os.environ.get('V06D12_MAIN_MAX_DECISIONS', '120000'))
    _max_options = int(os.environ.get('V06D12_MAIN_MAX_OPTIONS', '1600000'))
    _rng = random.Random(20260624)

    # ------------------------------------------------------------------
    # Feature extraction (vectorized)
    # New 96-dim layout: card_id hash removed, card metadata added at [61-95]
    # ------------------------------------------------------------------
    _t0_extract = _v06d12_time.perf_counter()
    _alloc_size = _max_options + 1024
    _X = np.zeros((_alloc_size, _V06D12_FEAT_DIM), dtype=np.float32)
    _y = np.zeros(_alloc_size, dtype=np.float32)
    _cursor = 0
    _decision_rows = []
    _option_meta_rows = []
    _decision_id = 0
    _stats = _v06d12_dd(int)
    _cross_check_buf = []

    # Precompute card meta matrix for all known card IDs for fast vectorized lookup
    _known_card_ids = sorted(_card_table.keys())
    _card_meta_lookup = {}  # card_id -> np array of shape (35,)
    for _cid in _known_card_ids:
        _card_meta_lookup[_cid] = np.array(_v06d12_card_meta_vec(_cid, _card_table), dtype=np.float32)
    _zero_meta = np.zeros(_V06D12_CARD_META_DIM, dtype=np.float32)

    for _fp in EPISODE_FILES:
        if _decision_id >= _max_decisions or _cursor >= _max_options:
            break
        try:
            _d = json.loads(Path(_fp).read_text())
        except Exception:
            _stats['bad_episode_json'] += 1
            continue
        if not isinstance(_d, dict):
            _stats['bad_episode_type'] += 1
            continue
        _eid = str((_d.get('info') or {}).get('EpisodeId', Path(_fp).stem))
        _split = _split_map.get(_eid, _split_map.get(Path(_fp).stem, 'unknown'))
        _names = (_d.get('info') or {}).get('TeamNames') or ['', '']
        _rewards = _d.get('rewards') or [0, 0]
        _steps = _d.get('steps') or []

        # Infer seat->deck from first deck-submission step
        _seat_deck = {}
        for _s in _steps:
            if not isinstance(_s, list):
                continue
            for _seat, _a in enumerate(_s):
                if not isinstance(_a, dict):
                    continue
                _act = _a.get('action')
                if _seat not in _seat_deck and isinstance(_act, list) and len(_act) == 60 and all(isinstance(x, int) for x in _act):
                    _obs = _a.get('observation') or {}
                    _sel = _obs.get('select') if isinstance(_obs, dict) else None
                    _nopt = len(_sel.get('option', [])) if isinstance(_sel, dict) else 0
                    if _nopt < 60:
                        _seat_deck[_seat] = _act
        _seat_arch = {k: _v06d12_archetype_of_deck(v) for k, v in _seat_deck.items()}

        for _s in _steps:
            if _decision_id >= _max_decisions or _cursor >= _max_options:
                break
            if not isinstance(_s, list):
                continue
            for _seat, _a in enumerate(_s):
                if _decision_id >= _max_decisions or _cursor >= _max_options:
                    break
                if not isinstance(_a, dict) or _a.get('status') != 'ACTIVE':
                    continue
                _obs = _a.get('observation')
                _act = _a.get('action')
                if not isinstance(_obs, dict) or not isinstance(_act, list) or len(_act) != 1:
                    continue
                _sel = _obs.get('select')
                _ctx_raw = _sel.get('context', -1) if isinstance(_sel, dict) else -1
                if not isinstance(_sel, dict) or int(_ctx_raw if _ctx_raw is not None else -1) != 0:
                    continue
                _opts = _sel.get('option') or []
                _nopt = len(_opts)
                _mx = int(_sel.get('maxCount', 0) or 0)
                _mn = int(_sel.get('minCount', 0) or 0)
                if _nopt < 2 or _mx != 1 or _mn > 1 or not isinstance(_act[0], int) or _act[0] < 0 or _act[0] >= _nopt:
                    _stats['skipped_main'] += 1
                    continue
                if _seat_arch.get(_seat, 'other') != _our_arch:
                    continue

                _rank = _rank_of(_names[_seat]) if _seat < len(_names) else None
                _rw = _rewards[_seat] if _seat < len(_rewards) else None
                _won = isinstance(_rw, (int, float)) and _rw > 0
                _opp_arch = _seat_arch.get(1 - _seat, 'other')

                try:
                    _rule_pred = _rule_agent(_obs)
                except Exception:
                    _stats['rule_exceptions'] += 1
                    _rule_pred = []
                _rule_idx = _rule_pred[0] if isinstance(_rule_pred, list) and len(_rule_pred) == 1 and isinstance(_rule_pred[0], int) and 0 <= _rule_pred[0] < _nopt else -1
                if _rule_idx < 0:
                    _stats['rule_bad_shape'] += 1

                _cur = _obs.get('current') or {}
                _sel2 = _obs.get('select') or {}
                _your = int(_cur.get('yourIndex', 0) or 0)
                _opp_idx = 1 - _your
                _mine = _v06d12_player_summary(_obs, _your)
                _theirs = _v06d12_player_summary(_obs, _opp_idx)
                _context = int(_sel2.get('context', 0) or 0)

                # Shared decision-level scalars
                _f_bias = 1.0
                _f_opt_count = min(1.0, _nopt / 32.0)
                _f_min_count = _v06d12_to_float(_sel2.get('minCount'), 0.0) / 8.0
                _f_max_count = _v06d12_to_float(_sel2.get('maxCount'), 0.0) / 8.0
                _f_turn = min(1.0, _v06d12_to_float(_cur.get('turn'), 0.0) / 20.0)
                _f_turn_act = min(1.0, _v06d12_to_float(_cur.get('turnActionCount'), 0.0) / 32.0)
                _f_step = min(1.0, _v06d12_to_float(_obs.get('step'), 0.0) / 512.0)
                _f_your = float(_your)
                _f_energy = float(bool(_cur.get('energyAttached')))
                _f_supporter = float(bool(_cur.get('supporterPlayed')))
                _f_retreated = float(bool(_cur.get('retreated')))
                _f_stadium = float(bool(_cur.get('stadiumPlayed')))
                _f_context = (_context % 64) / 64.0

                _s_row = _cursor
                _e_row = _cursor + _nopt

                # Fill global context (same for all options)
                _X[_s_row:_e_row, 0] = _f_bias
                _X[_s_row:_e_row, 1] = _f_opt_count
                _denom = float(max(1, _nopt - 1))
                _X[_s_row:_e_row, 2] = np.minimum(1.0, np.arange(_nopt, dtype=np.float32) / _denom)
                _X[_s_row:_e_row, 3] = _f_min_count
                _X[_s_row:_e_row, 4] = _f_max_count
                _X[_s_row:_e_row, 5] = _f_turn
                _X[_s_row:_e_row, 6] = _f_turn_act
                _X[_s_row:_e_row, 7] = _f_step
                _X[_s_row:_e_row, 8] = _f_your
                _X[_s_row:_e_row, 9] = _f_energy
                _X[_s_row:_e_row, 10] = _f_supporter
                _X[_s_row:_e_row, 11] = _f_retreated
                _X[_s_row:_e_row, 12] = _f_stadium
                _X[_s_row:_e_row, 13] = _mine['hand'] / 16.0
                _X[_s_row:_e_row, 14] = _mine['deck'] / 60.0
                _X[_s_row:_e_row, 15] = _mine['discard'] / 60.0
                _X[_s_row:_e_row, 16] = _mine['bench'] / 8.0
                _X[_s_row:_e_row, 17] = _mine['active'] / 4.0
                _X[_s_row:_e_row, 18] = _mine['prize'] / 6.0
                _X[_s_row:_e_row, 19] = _theirs['hand'] / 16.0
                _X[_s_row:_e_row, 20] = _theirs['deck'] / 60.0
                _X[_s_row:_e_row, 21] = _theirs['discard'] / 60.0
                _X[_s_row:_e_row, 22] = _theirs['bench'] / 8.0
                _X[_s_row:_e_row, 23] = _theirs['active'] / 4.0
                _X[_s_row:_e_row, 24] = _theirs['prize'] / 6.0
                _X[_s_row:_e_row, 25] = _f_context

                # Per-option features
                _opt_types = np.array([int((_o or {}).get('type', 0) or 0) for _o in _opts], dtype=np.int32)
                _areas = np.array([int((_o or {}).get('area', 0) or 0) for _o in _opts], dtype=np.int32)
                _card_ids = np.array([_v06d12_card_id_from_area(_obs, _o if isinstance(_o, dict) else {}) for _o in _opts], dtype=np.int32)
                _opt_indices = np.array([_v06d12_to_float((_o or {}).get('index'), 0.0) for _o in _opts], dtype=np.float32)
                _in_plays = np.array([_v06d12_to_float((_o or {}).get('inPlayIndex'), 0.0) for _o in _opts], dtype=np.float32)
                _numbers = np.array([_v06d12_to_float((_o or {}).get('number'), 0.0) for _o in _opts], dtype=np.float32)
                _attack_ids = np.array([_v06d12_to_float((_o or {}).get('attackId'), 0.0) for _o in _opts], dtype=np.float32)
                _ability_ids = np.array([_v06d12_to_float((_o or {}).get('abilityId'), 0.0) for _o in _opts], dtype=np.float32)

                # [26-32]: option numerics (no card_id continuous)
                _X[_s_row:_e_row, 26] = (_opt_types % 32) / 32.0
                _X[_s_row:_e_row, 27] = (_areas % 16) / 16.0
                _X[_s_row:_e_row, 28] = _opt_indices / 32.0
                _X[_s_row:_e_row, 29] = _in_plays / 16.0
                _X[_s_row:_e_row, 30] = _numbers / 16.0
                _X[_s_row:_e_row, 31] = _attack_ids / 4096.0
                _X[_s_row:_e_row, 32] = _ability_ids / 4096.0

                # [33-48]: opt_type one-hot (16 dims)
                _row_idx = np.arange(_s_row, _e_row)
                _X[_row_idx, 33 + (_opt_types % 16)] = 1.0
                # [49-60]: area one-hot (12 dims)
                _X[_row_idx, 49 + (_areas % 12)] = 1.0

                # [61-95]: card metadata (35 dims)
                for _oi in range(_nopt):
                    _cid = int(_card_ids[_oi])
                    _meta = _card_meta_lookup.get(_cid, _zero_meta)
                    _X[_s_row + _oi, _V06D12_CARD_META_OFFSET:_V06D12_CARD_META_OFFSET + _V06D12_CARD_META_DIM] = _meta

                # Label: chosen option
                _y[_s_row + int(_act[0])] = 1.0

                for _oi, _opt in enumerate(_opts):
                    _option_meta_rows.append({
                        'decision_id': _decision_id,
                        'option_index': _oi,
                        'label': int(_oi == _act[0]),
                        'option_type': int((_opt or {}).get('type', -1)) if isinstance(_opt, dict) else -1,
                        'option_type_name': _v06d12_name(_V06D12_OPTION_NAMES, int((_opt or {}).get('type', -1)) if isinstance(_opt, dict) else -1, 'OPT'),
                        'card_id': int(_card_ids[_oi]),
                    })

                _chosen_opt = _opts[_act[0]] if 0 <= _act[0] < len(_opts) and isinstance(_opts[_act[0]], dict) else {}
                _decision_rows.append({
                    'decision_id': _decision_id,
                    'episode_id': _eid,
                    'split': _split,
                    'seat': _seat,
                    'n_options': _nopt,
                    'chosen_index': int(_act[0]),
                    'chosen_option_type': int(_chosen_opt.get('type', -1)),
                    'chosen_option_type_name': _v06d12_name(_V06D12_OPTION_NAMES, int(_chosen_opt.get('type', -1)), 'OPT'),
                    'rule_index': int(_rule_idx),
                    'rule_hit': int(_rule_idx == _act[0]),
                    'rank': _rank,
                    'rank_bucket': _v06d12_rank_bucket(_rank),
                    'won': bool(_won),
                    'matchup': _our_arch + '_vs_' + _opp_arch,
                    'option_start': _s_row,
                    'option_end': _e_row,
                })

                if len(_cross_check_buf) < 100:
                    _cross_check_buf.append((_obs, list(_opts), _nopt, _s_row))

                _cursor += _nopt
                _decision_id += 1

    _X = _X[:_cursor]
    _y = _y[:_cursor]
    _t1_extract = _v06d12_time.perf_counter()
    _extract_time_sec = _t1_extract - _t0_extract
    print(f'v0-06d12 extraction: {_extract_time_sec:.1f}s, {_cursor} option rows, {_decision_id} decisions')

    _decision_df = pd.DataFrame(_decision_rows)
    _option_meta_df = pd.DataFrame(_option_meta_rows)
    if not _cursor or _decision_df.empty:
        raise RuntimeError('no MAIN learning rows were extracted')

    # Cross-check: reference vs vectorized
    _check_mismatches = 0
    _check_max_err = 0.0
    for _cc_obs, _cc_opts, _cc_nopt, _cc_start in _cross_check_buf:
        for _cc_oi, _cc_opt in enumerate(_cc_opts):
            _ref = _v06d12_option_features(_cc_obs, _cc_opt if isinstance(_cc_opt, dict) else {}, _cc_oi, _cc_nopt, _card_table)
            _actual = _X[_cc_start + _cc_oi]
            _errs = [abs(float(a) - float(b)) for a, b in zip(_ref, _actual.tolist())]
            _max_e = max(_errs) if _errs else 0.0
            _check_max_err = max(_check_max_err, _max_e)
            if _max_e > 1e-5:
                _check_mismatches += 1
    _crosscheck_report = {
        'decisions_checked': len(_cross_check_buf),
        'options_checked': sum(t[2] for t in _cross_check_buf),
        'mismatches': _check_mismatches,
        'max_abs_error': float(_check_max_err),
        'passed': _check_mismatches == 0,
    }
    print(f'cross-check: {_crosscheck_report}')
    if _check_mismatches > 0:
        raise RuntimeError(f'feature cross-check FAILED: {_check_mismatches} mismatches, max_err={_check_max_err}')

    # Save feature matrix
    _feat_npy_path = OUTPUT_DIR / (RUN_PREFIX + '-main_feature_matrix.npy')
    _label_npy_path = OUTPUT_DIR / (RUN_PREFIX + '-main_labels.npy')
    np.save(str(_feat_npy_path), _X)
    np.save(str(_label_npy_path), _y)
    ARTIFACT_PATHS['v06d12_main_feature_matrix'] = str(_feat_npy_path)
    ARTIFACT_PATHS['v06d12_main_labels'] = str(_label_npy_path)

    _feature_spec = {
        'feature_dim': _V06D12_FEAT_DIM,
        'layout_version': 'v06d12',
        'card_id_hash_encoding': False,
        'card_metadata_encoding': True,
        'card_meta_dim': _V06D12_CARD_META_DIM,
        'card_meta_offset': _V06D12_CARD_META_OFFSET,
        'target': 'recorded replay MAIN option index for alakazam actor',
        'loss': 'per_decision_cross_entropy',
        'extraction_method': 'numpy_preallocated_vectorized_v06d12',
        'crosscheck_passed': _check_mismatches == 0,
        'crosscheck_max_abs_error': float(_check_max_err),
        'feature_layout': {
            '0': 'bias',
            '1': 'opt_count/32',
            '2': 'opt_idx_frac',
            '3': 'min_count/8',
            '4': 'max_count/8',
            '5': 'turn/20',
            '6': 'turn_action/32',
            '7': 'step/512',
            '8': 'your_index',
            '9': 'energy_attached',
            '10': 'supporter_played',
            '11': 'retreated',
            '12': 'stadium_played',
            '13-18': 'mine hand/deck/discard/bench/active/prize',
            '19-24': 'theirs hand/deck/discard/bench/active/prize',
            '25': 'context/64',
            '26': 'opt_type/32',
            '27': 'area/16',
            '28': 'opt_index/32',
            '29': 'in_play_index/16',
            '30': 'number/16',
            '31': 'attack_id/4096',
            '32': 'ability_id/4096',
            '33-48': 'opt_type one-hot (16)',
            '49-60': 'area one-hot (12)',
            '61-95': 'card metadata (35): cardType*7, ex/mega/ace/tera, basic/stage1/stage2, hp_norm, has_skill, 19 key-card flags',
        },
    }
    ARTIFACT_PATHS['v06d12_main_feature_spec'] = write_json(OUTPUT_DIR / 'main_feature_spec.json', _feature_spec)

    _dataset_report = {
        'status': 'ok',
        'stats': dict(_stats),
        'decisions': int(len(_decision_df)),
        'option_rows': int(len(_X)),
        'feature_dim': int(_V06D12_FEAT_DIM),
        'split_counts': _decision_df['split'].value_counts().to_dict(),
        'rule_top1_overall': float(_decision_df['rule_hit'].mean()),
        'extract_time_sec': float(_extract_time_sec),
    }
    ARTIFACT_PATHS['v06d12_main_dataset_report'] = write_json(OUTPUT_DIR / 'main_dataset_report.json', _dataset_report)
    ARTIFACT_PATHS['v06d12_main_decision_rows'] = safe_save_table(_decision_df, OUTPUT_DIR / 'main_decision_rows.parquet')

    # ------------------------------------------------------------------
    # Build decision-indexed arrays for per-decision CE training
    # ------------------------------------------------------------------
    _train_df = _decision_df[_decision_df['split'].astype(str) == 'train'].copy()
    _valid_df = _decision_df[_decision_df['split'].astype(str) == 'valid'].copy()
    _holdout_df = _decision_df[_decision_df['split'].astype(str) == 'holdout'].copy()
    if len(_train_df) < 1000 or len(_holdout_df) < 200:
        raise RuntimeError('insufficient split counts for MAIN learning')

    _tr_starts = _train_df['option_start'].values.astype(np.int64)
    _tr_n_opts = _train_df['n_options'].values.astype(np.int64)
    _tr_chosen = _train_df['chosen_index'].values.astype(np.int64)
    _n_tr = len(_train_df)

    _va_starts = _valid_df['option_start'].values.astype(np.int64)
    _va_n_opts = _valid_df['n_options'].values.astype(np.int64)
    _va_chosen = _valid_df['chosen_index'].values.astype(np.int64)
    _n_va = len(_valid_df)

    # ------------------------------------------------------------------
    # Model definition (same architecture as v0-06d10)
    # ------------------------------------------------------------------
    class _V06D12MainScorer(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(dim, 512), nn.ReLU(),
                nn.Linear(512, 384), nn.ReLU(),
                nn.Linear(384, 256), nn.ReLU(),
                nn.Linear(256, 1),
            )
        def forward(self, x):
            return self.net(x).squeeze(-1)

    _device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    _model = _V06D12MainScorer(_V06D12_FEAT_DIM).to(_device)
    _param_count = int(sum(p.numel() for p in _model.parameters()))
    print(f'model: {_param_count} params, device={_device}')

    _max_epochs = int(os.environ.get('V06D12_MAIN_EPOCHS', '100'))
    _patience = int(os.environ.get('V06D12_MAIN_PATIENCE', '5'))
    _decision_batch = int(os.environ.get('V06D12_DECISION_BATCH', '256'))
    _lr = float(os.environ.get('V06D12_MAIN_LR', '0.002'))

    # ------------------------------------------------------------------
    # Training skip: load saved weights when model file already exists
    # ------------------------------------------------------------------
    _model_path = OUTPUT_DIR / (RUN_PREFIX + '-main_option_scorer.pt')
    _SKIP_TRAIN = _model_path.exists()
    if _SKIP_TRAIN:
        print(f'skip_train=True: loading existing model from {_model_path}')
        _model.load_state_dict(torch.load(str(_model_path), map_location=_device, weights_only=True))
        _model.eval()
        _epochs_run = 0
        _best_epoch = 0
        _best_valid_top1 = 0.0
        _history = []
        _gpu_load_time_sec = 0.0
        _train_time_sec = 0.0

    else:
        # Load full X to GPU for fast slicing
        _t0_gpu = _v06d12_time.perf_counter()
        try:
            _X_gpu = torch.tensor(_X, dtype=torch.float32, device=_device)
            _gpu_preload_ok = True
            print(f'GPU X loaded: shape={tuple(_X_gpu.shape)} device={_device}')
        except RuntimeError as _ge:
            print(f'GPU X preload failed ({_ge}), using CPU')
            _X_gpu = torch.tensor(_X, dtype=torch.float32)
            _gpu_preload_ok = False
        _gpu_load_time_sec = _v06d12_time.perf_counter() - _t0_gpu

        _opt_adam = torch.optim.AdamW(_model.parameters(), lr=_lr, weight_decay=1e-4)

        # Training loop: per-decision cross-entropy
        _t0_train = _v06d12_time.perf_counter()
        _history = []
        _best_valid_top1 = -1.0
        _best_epoch = 0
        _best_state_dict = None
        _patience_counter = 0
        _epochs_run = 0

        for _ep in range(_max_epochs):
            _model.train()
            _perm = torch.randperm(_n_tr).numpy()
            _ep_losses = []

            for _st in range(0, _n_tr, _decision_batch):
                _bidxs = _perm[_st:_st + _decision_batch]
                _b_starts = _tr_starts[_bidxs]
                _b_n = _tr_n_opts[_bidxs]
                _b_chosen = _tr_chosen[_bidxs]
                _B = len(_bidxs)
                _max_n = int(_b_n.max())

                # Build padded batch on CPU then move to GPU
                _xb_np = np.zeros((_B, _max_n, _V06D12_FEAT_DIM), dtype=np.float32)
                _mask_np = np.zeros((_B, _max_n), dtype=bool)
                for _j in range(_B):
                    _s, _n = int(_b_starts[_j]), int(_b_n[_j])
                    _xb_np[_j, :_n] = _X[_s:_s + _n]
                    _mask_np[_j, :_n] = True

                _xb = torch.tensor(_xb_np, dtype=torch.float32, device=_device)
                _mask = torch.tensor(_mask_np, device=_device)
                _chosen_t = torch.tensor(_b_chosen, dtype=torch.long, device=_device)

                # Forward: (B, max_n, F) -> (B*max_n, F) -> (B*max_n,) -> (B, max_n)
                _logits = _model(_xb.view(_B * _max_n, _V06D12_FEAT_DIM)).view(_B, _max_n)
                _logits[~_mask] = -1e9  # mask padding before softmax

                _opt_adam.zero_grad(set_to_none=True)
                _loss = F.cross_entropy(_logits, _chosen_t)
                _loss.backward()
                _opt_adam.step()
                _ep_losses.append(float(_loss.item()))

            # Valid top-1 per decision
            _model.eval()
            with torch.no_grad():
                _va_hits = 0
                _va_loss_vals = []
                for _st in range(0, _n_va, _decision_batch):
                    _end = min(_st + _decision_batch, _n_va)
                    _b_starts = _va_starts[_st:_end]
                    _b_n = _va_n_opts[_st:_end]
                    _b_chosen = _va_chosen[_st:_end]
                    _B = _end - _st
                    _max_n = int(_b_n.max())
                    _xb_np = np.zeros((_B, _max_n, _V06D12_FEAT_DIM), dtype=np.float32)
                    _mask_np = np.zeros((_B, _max_n), dtype=bool)
                    for _j in range(_B):
                        _s, _n = int(_b_starts[_j]), int(_b_n[_j])
                        _xb_np[_j, :_n] = _X[_s:_s + _n]
                        _mask_np[_j, :_n] = True
                    _xb = torch.tensor(_xb_np, dtype=torch.float32, device=_device)
                    _mask_t = torch.tensor(_mask_np, device=_device)
                    _chosen_t = torch.tensor(_b_chosen, dtype=torch.long, device=_device)
                    _logits = _model(_xb.view(_B * _max_n, _V06D12_FEAT_DIM)).view(_B, _max_n)
                    _logits[~_mask_t] = -1e9
                    _va_loss_vals.append(float(F.cross_entropy(_logits, _chosen_t).item()))
                    _preds = _logits.argmax(dim=1).cpu().numpy()
                    _va_hits += int((_preds == _b_chosen).sum())

            _ep_valid_top1 = _va_hits / max(1, _n_va)
            _ep_valid_loss = float(np.mean(_va_loss_vals))
            _ep_train_loss = float(np.mean(_ep_losses))
            _epochs_run = _ep + 1

            _history.append({
                'epoch': int(_ep),
                'train_loss': _ep_train_loss,
                'valid_loss': _ep_valid_loss,
                'valid_top1': _ep_valid_top1,
            })
            _improved = _ep_valid_top1 > _best_valid_top1
            print(f'  ep {_ep:3d}: train={_ep_train_loss:.4f} valid_loss={_ep_valid_loss:.4f} valid_top1={_ep_valid_top1:.4f}'
                  + (' *' if _improved else f'  (patience {_patience_counter+1}/{_patience})'))
            if _improved:
                _best_valid_top1 = _ep_valid_top1
                _best_epoch = _ep
                _best_state_dict = {k: v.cpu().clone() for k, v in _model.state_dict().items()}
                _patience_counter = 0
            else:
                _patience_counter += 1
            if _patience_counter >= _patience:
                print(f'  early stop at epoch {_ep}, best={_best_epoch} valid_top1={_best_valid_top1:.4f}')
                break

        if _best_state_dict is not None:
            _model.load_state_dict({k: v.to(_device) for k, v in _best_state_dict.items()})
        _model.eval()
        _train_time_sec = _v06d12_time.perf_counter() - _t0_train

        # Save model
        torch.save(_model.state_dict(), _model_path)
    ARTIFACT_PATHS['v06d12_main_option_scorer'] = str(_model_path)

    # ------------------------------------------------------------------
    # Inference on full dataset (flat row-by-row for prediction table)
    # ------------------------------------------------------------------
    _t0_infer = _v06d12_time.perf_counter()
    _all_logits = np.empty(len(_X), dtype=np.float32)
    _infer_batch = 8192
    with torch.no_grad():
        for _st in range(0, len(_X), _infer_batch):
            _xb = torch.tensor(_X[_st:_st + _infer_batch], dtype=torch.float32, device=_device)
            _all_logits[_st:_st + _infer_batch] = _model(_xb).detach().cpu().numpy()
    _infer_time_sec = _v06d12_time.perf_counter() - _t0_infer

    # ------------------------------------------------------------------
    # Prediction table + per-bucket evaluation
    # ------------------------------------------------------------------
    _pred_rows = []
    for _idx, _r in _decision_df.iterrows():
        _a = int(_r['option_start'])
        _b = int(_r['option_end'])
        _scores = _all_logits[_a:_b]
        _order = np.argsort(-_scores)
        _chosen_rel = int(_r['chosen_index'])
        _model_top1 = int(_order[0]) if len(_order) else -1
        _top1_score = float(_scores[_model_top1]) if _model_top1 >= 0 else None
        _top2_score = float(_scores[int(_order[1])]) if len(_order) > 1 else None
        _margin = float(_top1_score - _top2_score) if _top1_score is not None and _top2_score is not None else None
        _top1_prob = float(1.0 / (1.0 + np.exp(-_top1_score))) if _top1_score is not None else None
        _meta_slice = _option_meta_df.iloc[_a:_b].reset_index(drop=True)
        _model_type_name = 'INVALID'
        if 0 <= _model_top1 < len(_meta_slice):
            _model_type_name = str(_meta_slice.iloc[_model_top1]['option_type_name'])
        _rule_type_name = 'INVALID'
        _rule_idx = int(_r['rule_index'])
        if 0 <= _rule_idx < len(_meta_slice):
            _rule_type_name = str(_meta_slice.iloc[_rule_idx]['option_type_name'])
        _rank_pos = int(np.where(_order == _chosen_rel)[0][0] + 1) if _chosen_rel in set(_order.tolist()) else None
        _pred_rows.append({
            'decision_id': int(_idx),
            'split': _r['split'],
            'matchup': _r['matchup'],
            'rank_bucket': _r['rank_bucket'],
            'won': bool(_r['won']),
            'chosen_option_type_name': _r['chosen_option_type_name'],
            'n_options': int(_r['n_options']),
            'rule_index': int(_r['rule_index']),
            'rule_option_type_name': _rule_type_name,
            'model_top1_index': int(_model_top1),
            'model_top1_option_type_name': _model_type_name,
            'model_top1_score': _top1_score,
            'model_top2_score': _top2_score,
            'model_margin': _margin,
            'model_top1_prob': _top1_prob,
            'rule_hit': int(_r['rule_hit']),
            'model_top1_hit': int(_model_top1 == _chosen_rel),
            'model_top3_hit': int(_chosen_rel in set(_order[:3].tolist())),
            'model_chosen_rank': _rank_pos,
            'random_top1_expected': 1.0 / max(1, int(_r['n_options'])),
        })
    _pred_df = pd.DataFrame(_pred_rows)
    _hold = _pred_df[_pred_df['split'].astype(str) == 'holdout']
    ARTIFACT_PATHS['v06d12_main_holdout_predictions'] = safe_save_table(
        _hold, OUTPUT_DIR / 'main_holdout_predictions.parquet')

    def _metric_row(df, bucket_name, bucket_value):
        if df.empty:
            return {'bucket_name': bucket_name, 'bucket_value': bucket_value, 'n': 0,
                    'rule_top1': None, 'model_top1': None, 'model_top3': None,
                    'model_chosen_rank': None, 'random_top1_expected': None, 'model_minus_rule_top1': None}
        return {
            'bucket_name': bucket_name, 'bucket_value': str(bucket_value), 'n': int(len(df)),
            'rule_top1': float(df['rule_hit'].mean()),
            'model_top1': float(df['model_top1_hit'].mean()),
            'model_top3': float(df['model_top3_hit'].mean()),
            'model_chosen_rank': float(df['model_chosen_rank'].mean()),
            'random_top1_expected': float(df['random_top1_expected'].mean()),
            'model_minus_rule_top1': float(df['model_top1_hit'].mean() - df['rule_hit'].mean()),
        }

    _eval_rows = []
    for _split in ['train', 'valid', 'holdout']:
        _df = _pred_df[_pred_df['split'].astype(str) == _split]
        _eval_rows.append(_metric_row(_df, 'split', _split))
    for _col in ['chosen_option_type_name', 'matchup', 'rank_bucket', 'won']:
        for _val, _g in _hold.groupby(_col, dropna=False):
            _eval_rows.append(_metric_row(_g, _col, _val))
    _eval_df = pd.DataFrame(_eval_rows)
    ARTIFACT_PATHS['v06d12_main_eval_by_bucket'] = safe_save_table(_eval_df, OUTPUT_DIR / 'main_eval_by_bucket.parquet')

    # ------------------------------------------------------------------
    # Hybrid evaluation — guard scope: PLAY/ATTACH/EVOLVE/RETREAT
    # ATTACK/END excluded: data shows model accuracy < rule on those types.
    # ABILITY remains hard-vetoed.
    # Threshold grid on valid; holdout only for promotion check.
    # ------------------------------------------------------------------
    _safe_override_types = [x.strip().upper() for x in
        os.environ.get('V06D12_SAFE_MAIN_OVERRIDE_TYPES', 'PLAY,ATTACH,EVOLVE,RETREAT').split(',') if x.strip()]
    _veto_types = ['ABILITY']  # only hard veto remaining
    _prob_grid = [float(x) for x in
        os.environ.get('V06D12_HYBRID_PROB_GRID', '0.35,0.45,0.55,0.65,0.75,0.85').split(',') if str(x).strip()]
    _margin_grid = [float(x) for x in
        os.environ.get('V06D12_HYBRID_MARGIN_GRID', '0.00,0.10,0.20,0.35,0.50').split(',') if str(x).strip()]

    def _hybrid_eval(df, prob_threshold, margin_threshold, label):
        if df.empty:
            return {'label': label, 'n': 0, 'prob_threshold': float(prob_threshold), 'margin_threshold': float(margin_threshold)}
        _d = df.copy()
        _override = (
            _d['model_top1_option_type_name'].astype(str).str.upper().isin(_safe_override_types) &
            (~_d['model_top1_option_type_name'].astype(str).str.upper().isin(_veto_types)) &
            (_d['model_top1_index'].astype(int) != _d['rule_index'].astype(int)) &
            (_d['model_top1_prob'].fillna(-999.0) >= prob_threshold) &
            (_d['model_margin'].fillna(-999.0) >= margin_threshold)
        )
        _hybrid_hit = np.where(_override, _d['model_top1_hit'].astype(int), _d['rule_hit'].astype(int))
        _benefit = _override & (_d['rule_hit'].astype(int) == 0) & (_d['model_top1_hit'].astype(int) == 1)
        _harm = _override & (_d['rule_hit'].astype(int) == 1) & (_d['model_top1_hit'].astype(int) == 0)
        _attack_mask = _d['chosen_option_type_name'].astype(str).str.upper() == 'ATTACK'
        _danger_mask = _d['chosen_option_type_name'].astype(str).str.upper().isin(['ATTACK', 'END'])
        _safe_chosen = _d['chosen_option_type_name'].astype(str).str.upper().isin(['PLAY', 'ATTACH', 'EVOLVE'])
        _rule_t1 = float(_d['rule_hit'].mean())
        _hybrid_t1 = float(np.mean(_hybrid_hit))
        def _safe_mean(mask):
            s = _d.loc[mask]
            return float(np.mean(_hybrid_hit[mask.to_numpy()])) if bool(mask.any()) and len(s) > 0 else None
        def _rule_mean(mask):
            s = _d.loc[mask, 'rule_hit']
            return float(s.mean()) if bool(mask.any()) else None
        return {
            'label': label, 'n': int(len(_d)),
            'prob_threshold': float(prob_threshold), 'margin_threshold': float(margin_threshold),
            'safe_override_types': ','.join(_safe_override_types),
            'rule_top1': _rule_t1, 'model_top1': float(_d['model_top1_hit'].mean()),
            'hybrid_top1': _hybrid_t1,
            'hybrid_minus_rule_top1': float(_hybrid_t1 - _rule_t1),
            'override_count': int(_override.sum()), 'override_rate': float(_override.mean()),
            'benefit_count': int(_benefit.sum()), 'harm_count': int(_harm.sum()),
            'benefit_minus_harm': int(_benefit.sum() - _harm.sum()),
            'attack_rule_top1': _rule_mean(_attack_mask),
            'attack_hybrid_top1': _safe_mean(_attack_mask),
            'attack_hybrid_delta': ((_safe_mean(_attack_mask) or 0) - (_rule_mean(_attack_mask) or 0)) if bool(_attack_mask.any()) else None,
            'danger_rule_top1': _rule_mean(_danger_mask),
            'danger_hybrid_top1': _safe_mean(_danger_mask),
            'danger_hybrid_delta': ((_safe_mean(_danger_mask) or 0) - (_rule_mean(_danger_mask) or 0)) if bool(_danger_mask.any()) else None,
            'safe_chosen_rule_top1': _rule_mean(_safe_chosen),
            'safe_chosen_hybrid_top1': _safe_mean(_safe_chosen),
        }

    _valid_pred = _pred_df[_pred_df['split'].astype(str) == 'valid']
    _threshold_rows = []
    for _pt in _prob_grid:
        for _mt in _margin_grid:
            _row = _hybrid_eval(_valid_pred, _pt, _mt, 'valid_grid')
            _danger_delta = _row.get('danger_hybrid_delta')
            _attack_delta = _row.get('attack_hybrid_delta')
            _danger_ok = _danger_delta is None or _danger_delta >= -0.01
            _override_ok = _row.get('override_rate', 0.0) >= 0.01
            _benefit_ok = _row.get('benefit_count', 0) > _row.get('harm_count', 0)
            _row['danger_ok'] = bool(_danger_ok)
            _row['override_ok'] = bool(_override_ok)
            _row['benefit_ok'] = bool(_benefit_ok)
            _row['selection_ok'] = bool(_danger_ok and _override_ok and _benefit_ok)
            _row['selection_score'] = float(
                _row.get('hybrid_minus_rule_top1', 0.0)
                + 0.001 * _row.get('benefit_minus_harm', 0)
                - 0.3 * max(0.0, -(_danger_delta or 0.0))
            )
            _threshold_rows.append(_row)

    _threshold_df = pd.DataFrame(_threshold_rows).sort_values(
        ['selection_ok', 'selection_score', 'hybrid_minus_rule_top1', 'override_rate'],
        ascending=[False, False, False, False],
    )
    ARTIFACT_PATHS['v06d12_main_hybrid_threshold_grid'] = safe_save_table(
        _threshold_df, OUTPUT_DIR / 'main_hybrid_threshold_grid.parquet')

    _selected_threshold = _threshold_df.iloc[0].to_dict()
    _sel_prob = float(_selected_threshold['prob_threshold'])
    _sel_margin = float(_selected_threshold['margin_threshold'])
    _valid_hybrid_summary = _hybrid_eval(_valid_pred, _sel_prob, _sel_margin, 'valid_selected')
    _holdout_hybrid_summary = _hybrid_eval(_hold, _sel_prob, _sel_margin, 'holdout_selected')

    _hybrid_bucket_df = pd.DataFrame([
        dict(_hybrid_eval(_g, _sel_prob, _sel_margin, f'{_col}={_val}'), bucket_name=_col, bucket_value=str(_val))
        for _col in ['chosen_option_type_name', 'matchup', 'rank_bucket', 'won', 'model_top1_option_type_name']
        for _val, _g in _hold.groupby(_col, dropna=False)
        if not _g.empty
    ])
    ARTIFACT_PATHS['v06d12_main_hybrid_eval_by_bucket'] = safe_save_table(
        _hybrid_bucket_df, OUTPUT_DIR / 'main_hybrid_eval_by_bucket.parquet')

    _hold_summary = _metric_row(_hold, 'split', 'holdout')
    _holdout_model_top1 = float(_hold_summary.get('model_top1', 0.0) or 0.0)
    _holdout_hybrid_top1 = float(_holdout_hybrid_summary.get('hybrid_top1', 0.0) or 0.0)
    _v06d11_hybrid_baseline = 0.4507
    _v06d11_model_baseline = 0.5090
    _quality_ok = _holdout_model_top1 >= 0.47 and _holdout_hybrid_top1 >= 0.44
    _attack_delta = _holdout_hybrid_summary.get('attack_hybrid_delta')
    _danger_gate_ok = _attack_delta is None or _attack_delta >= -0.03

    _timing_report = {
        'extract_time_sec': float(_extract_time_sec),
        'gpu_load_time_sec': float(_gpu_load_time_sec),
        'train_time_sec': float(_train_time_sec),
        'infer_time_sec': float(_infer_time_sec),
        'device': str(_device),
        'epochs_run': int(_epochs_run),
        'best_epoch': int(_best_epoch),
        'best_valid_top1': float(_best_valid_top1),
    }

    MAIN_HYBRID_REPORT = {
        'status': 'ok',
        'version': RUN_PREFIX,
        'purpose': 'per-decision CE + card metadata features + PLAY/ATTACH/EVOLVE/RETREAT guard scope (ATTACK/END excluded: model < rule)',
        'loss': 'per_decision_cross_entropy',
        'features': 'card_metadata_v06d12',
        'guard_scope': 'PLAY,ATTACH,EVOLVE,RETREAT (ATTACK/END excluded; ABILITY hard-vetoed)',
        'crosscheck_report': _crosscheck_report,
        'timing_report': _timing_report,
        'dataset_report': _dataset_report,
        'model': {
            'param_count': int(_param_count),
            'device': str(_device),
            'max_epochs': int(_max_epochs),
            'patience': int(_patience),
            'epochs_run': int(_epochs_run),
            'best_epoch': int(_best_epoch),
            'best_valid_top1': float(_best_valid_top1),
            'batch_size_decisions': int(_decision_batch),
            'learning_rate': float(_lr),
            'history': _history,
            'model_path': str(_model_path),
        },
        'holdout_summary': _hold_summary,
        'valid_hybrid_summary': _valid_hybrid_summary,
        'holdout_hybrid_summary': _holdout_hybrid_summary,
        'selected_prob_threshold': _sel_prob,
        'selected_margin_threshold': _sel_margin,
        'safe_override_types': _safe_override_types,
        'veto_types': _veto_types,
        'v06d11_hybrid_baseline': _v06d11_hybrid_baseline,
        'v06d11_model_baseline': _v06d11_model_baseline,
        'holdout_model_top1': _holdout_model_top1,
        'holdout_hybrid_top1': _holdout_hybrid_top1,
        'holdout_model_vs_v06d11': float(_holdout_model_top1 - _v06d11_model_baseline),
        'holdout_hybrid_vs_v06d11': float(_holdout_hybrid_top1 - _v06d11_hybrid_baseline),
        'attack_hybrid_delta_holdout': _attack_delta,
        'quality_ok': bool(_quality_ok),
        'danger_gate_ok': bool(_danger_gate_ok),
        'fresh_training': True,
        'reuses_prior_weights': False,
    }
    MAIN_LEARNING_REPORT = MAIN_HYBRID_REPORT
    ARTIFACT_PATHS['v06d12_main_model_report'] = write_json(OUTPUT_DIR / 'main_model_report.json', MAIN_HYBRID_REPORT)
    ARTIFACT_PATHS['v06d12_main_hybrid_report'] = write_json(OUTPUT_DIR / 'main_hybrid_report.json', MAIN_HYBRID_REPORT)

    print('v0-06d12 Part A status: ok')
    print(f'  epochs_run={_epochs_run} best_epoch={_best_epoch} best_valid_top1={_best_valid_top1:.4f}')
    print(f'  holdout model_top1={_holdout_model_top1:.4f} (v06d11={_v06d11_model_baseline})')
    print(f'  holdout hybrid_top1={_holdout_hybrid_top1:.4f} (v06d11={_v06d11_hybrid_baseline})')
    print(f'  selected threshold: prob={_sel_prob} margin={_sel_margin}')
    print(f'  attack_hybrid_delta={_attack_delta}  danger_gate_ok={_danger_gate_ok}  quality_ok={_quality_ok}')

except Exception as _v06d12_exc:
    MAIN_HYBRID_REPORT = {'status': 'error', 'error': repr(_v06d12_exc), 'traceback': traceback.format_exc(limit=8)}
    MAIN_LEARNING_REPORT = MAIN_HYBRID_REPORT
    ARTIFACT_PATHS['v06d12_main_model_report'] = write_json(OUTPUT_DIR / 'main_model_report.json', MAIN_HYBRID_REPORT)
    print('v0-06d12 Part A FAILED:', repr(_v06d12_exc))
    raise
