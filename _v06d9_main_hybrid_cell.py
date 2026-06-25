# =====================================================================
# v0-06d9 MAIN replay-supervised guarded hybrid
# Infrastructure speedup: NumPy vectorized feature extraction + GPU pre-loading.
# Same model architecture, feature definition, and hypothesis as v0-06d8.
# Cross-check enforced before training: vectorized features must match
# Python-loop features to within 1e-5 on a 100-decision sample.
# =====================================================================
import importlib.util as _v06d7_ilu
import math as _v06d7_math
import tarfile as _v06d7_tarfile
import time as _v06d9_time
from collections import defaultdict as _v06d7_dd

MAIN_HYBRID_REPORT = {'status': 'not_run'}
MAIN_LEARNING_REPORT = MAIN_HYBRID_REPORT

_V06D7_CONTEXT_NAMES = {
    0: 'MAIN', 1: 'SETUP_ACTIVE', 2: 'SETUP_BENCH', 3: 'SWITCH', 4: 'TO_ACTIVE',
    5: 'TO_BENCH', 6: 'TO_FIELD', 7: 'TO_HAND', 8: 'DISCARD', 9: 'TO_DECK',
    10: 'TO_DECK_BOTTOM', 11: 'TO_PRIZE', 12: 'NOT_MOVE', 13: 'DAMAGE_COUNTER',
    14: 'DAMAGE_COUNTER_ANY', 21: 'ATTACH_FROM', 22: 'ATTACH_TO', 30: 'DISCARD_ENERGY',
    31: 'TO_HAND_ENERGY', 33: 'SWITCH_ENERGY', 35: 'ATTACK', 37: 'EVOLVE',
    41: 'IS_FIRST', 42: 'MULLIGAN', 43: 'ACTIVATE',
}
_V06D7_OPTION_NAMES = {
    0: 'NUMBER', 1: 'YES', 2: 'NO', 3: 'CARD', 4: 'TOOL_CARD',
    5: 'ENERGY_CARD', 6: 'ENERGY', 7: 'PLAY', 8: 'ATTACH',
    9: 'EVOLVE', 10: 'ABILITY', 11: 'DISCARD', 12: 'RETREAT',
    13: 'ATTACK', 14: 'END', 15: 'SKILL', 16: 'SPECIAL_CONDITION',
}


def _v06d7_name(mapping, value, prefix):
    try:
        return mapping.get(int(value), prefix + '_' + str(value))
    except Exception:
        return prefix + '_' + str(value)


def _v06d7_norm_name(s):
    return re.sub(r'[^a-z0-9]', '', str(s).lower())


def _v06d7_rank_bucket(rank):
    if rank is None:
        return 'unranked'
    if rank <= 10:
        return 'top10'
    if rank <= 50:
        return 'top50'
    if rank <= 200:
        return 'top200'
    return 'other'


def _v06d7_archetype_of_deck(deck):
    s = set(int(x) for x in deck)
    if 743 in s:
        return 'alakazam'
    if 878 in s or 879 in s:
        return 'hop_control'
    if 678 in s or 673 in s:
        return 'lucario'
    return 'other'


def _v06d7_load_agent_from_archive(archive_path):
    with _v06d7_tarfile.open(archive_path) as t:
        src = t.extractfile('main.py').read().decode()
        deck = [int(x) for x in t.extractfile('deck.csv').read().decode().split() if str(x).strip()]
    mod_path = OUTPUT_DIR / (RUN_PREFIX + '-main_learning_rule_agent.py')
    mod_path.write_text(src, encoding='utf-8')
    spec = _v06d7_ilu.spec_from_file_location('v06d9_main_learning_rule_agent', str(mod_path))
    m = _v06d7_ilu.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m, deck


def _v06d7_to_float(x, default=0.0):
    try:
        if x is None:
            return float(default)
        return float(x)
    except Exception:
        return float(default)


def _v06d7_card_id_from_area(obs, option):
    try:
        area = int(option.get('area', 0))
        idx = int(option.get('index', -1))
        player = int(option.get('playerIndex', (obs.get('current') or {}).get('yourIndex', 0)))
        cur = obs.get('current') or {}
        players = cur.get('players') or []
        card = None
        if area == 1:
            select = obs.get('select') or {}
            deck_cards = select.get('deck') or []
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


def _v06d7_player_summary(obs, player_idx):
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


def _v06d7_option_features(obs, option, option_index, option_count):
    """Reference Python-loop feature extractor (used for cross-check only)."""
    cur = obs.get('current') or {}
    select = obs.get('select') or {}
    your = int(cur.get('yourIndex', 0) or 0)
    opp = 1 - your
    mine = _v06d7_player_summary(obs, your)
    theirs = _v06d7_player_summary(obs, opp)
    context = int(select.get('context', 0) or 0)
    opt_type = int(option.get('type', 0) or 0)
    area = int(option.get('area', 0) or 0)
    card_id = _v06d7_card_id_from_area(obs, option)
    row = [0.0] * 96
    row[0] = 1.0
    row[1] = min(1.0, option_count / 32.0)
    row[2] = min(1.0, option_index / max(1, option_count - 1))
    row[3] = _v06d7_to_float(select.get('minCount'), 0.0) / 8.0
    row[4] = _v06d7_to_float(select.get('maxCount'), 0.0) / 8.0
    row[5] = min(1.0, _v06d7_to_float(cur.get('turn'), 0.0) / 20.0)
    row[6] = min(1.0, _v06d7_to_float(cur.get('turnActionCount'), 0.0) / 32.0)
    row[7] = min(1.0, _v06d7_to_float(obs.get('step'), 0.0) / 512.0)
    row[8] = float(your)
    row[9] = float(bool(cur.get('energyAttached')))
    row[10] = float(bool(cur.get('supporterPlayed')))
    row[11] = float(bool(cur.get('retreated')))
    row[12] = float(bool(cur.get('stadiumPlayed')))
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
    row[25] = (context % 64) / 64.0
    row[26] = (opt_type % 32) / 32.0
    row[27] = (area % 16) / 16.0
    row[28] = (card_id % 4096) / 4096.0
    row[29] = _v06d7_to_float(option.get('index'), 0.0) / 32.0
    row[30] = _v06d7_to_float(option.get('inPlayIndex'), 0.0) / 16.0
    row[31] = _v06d7_to_float(option.get('number'), 0.0) / 16.0
    row[32] = _v06d7_to_float(option.get('attackId'), 0.0) / 4096.0
    row[33] = _v06d7_to_float(option.get('abilityId'), 0.0) / 4096.0
    row[34 + (opt_type % 16)] = 1.0
    row[50 + (area % 12)] = 1.0
    row[62 + (card_id % 32)] = 1.0
    row[94] = float(opt_type == 13)
    row[95] = float(opt_type == 14)
    return row


try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    torch.set_num_threads(int(os.environ.get('V06_TORCH_THREADS', '1')))

    _archive = WORKING_DIR / (RUN_PREFIX + '-submission.tar.gz')
    if not _archive.exists():
        raise FileNotFoundError('expected base submission archive not found: ' + str(_archive))
    _rule_mod, _our_deck = _v06d7_load_agent_from_archive(_archive)
    _rule_agent = _rule_mod.agent
    _our_arch = _v06d7_archetype_of_deck(_our_deck)

    _split_map = {}
    if 'EPISODE_SPLIT_DF' in globals() and isinstance(EPISODE_SPLIT_DF, pd.DataFrame) and not EPISODE_SPLIT_DF.empty:
        for _, r in EPISODE_SPLIT_DF.iterrows():
            _eid = str(r.get('episode_id', ''))
            _split_map[_eid] = str(r.get('split', 'unknown'))

    _rank_info = TOP200_LOOKUP if 'TOP200_LOOKUP' in globals() else {}
    def _rank_of(name):
        info = _rank_info.get(_v06d7_norm_name(name), {}) if isinstance(_rank_info, dict) else {}
        r = info.get('ranking') if isinstance(info, dict) else None
        return int(r) if r is not None else None

    _max_decisions = int(os.environ.get('V06D9_MAIN_MAX_DECISIONS', os.environ.get('V06D8_MAIN_MAX_DECISIONS', '120000')))
    _max_options = int(os.environ.get('V06D9_MAIN_MAX_OPTIONS', os.environ.get('V06D8_MAIN_MAX_OPTIONS', '1600000')))
    _rng = random.Random(20260624)

    # ------------------------------------------------------------------
    # Feature extraction: NumPy pre-allocation + vectorized per-decision
    # ------------------------------------------------------------------
    _t0_extract = _v06d9_time.perf_counter()

    _alloc_size = _max_options + 1024
    _X = np.zeros((_alloc_size, 96), dtype=np.float32)
    _y = np.zeros(_alloc_size, dtype=np.float32)
    _cursor = 0

    _decision_rows = []
    _option_meta_rows = []
    _decision_id = 0
    _stats = _v06d7_dd(int)
    _cross_check_buf = []  # (obs, opts, n_opts, row_start) for first 100 decisions

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
        _seat_arch = {k: _v06d7_archetype_of_deck(v) for k, v in _seat_deck.items()}

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
                _mine = _v06d7_player_summary(_obs, _your)
                _theirs = _v06d7_player_summary(_obs, _opp_idx)
                _context = int(_sel2.get('context', 0) or 0)

                # Shared scalars for this decision
                _f_bias = 1.0
                _f_opt_count = min(1.0, _nopt / 32.0)
                _f_min_count = _v06d7_to_float(_sel2.get('minCount'), 0.0) / 8.0
                _f_max_count = _v06d7_to_float(_sel2.get('maxCount'), 0.0) / 8.0
                _f_turn = min(1.0, _v06d7_to_float(_cur.get('turn'), 0.0) / 20.0)
                _f_turn_act = min(1.0, _v06d7_to_float(_cur.get('turnActionCount'), 0.0) / 32.0)
                _f_step = min(1.0, _v06d7_to_float(_obs.get('step'), 0.0) / 512.0)
                _f_your = float(_your)
                _f_energy = float(bool(_cur.get('energyAttached')))
                _f_supporter = float(bool(_cur.get('supporterPlayed')))
                _f_retreated = float(bool(_cur.get('retreated')))
                _f_stadium = float(bool(_cur.get('stadiumPlayed')))
                _f_context = (_context % 64) / 64.0

                _s_row = _cursor
                _e_row = _cursor + _nopt

                # Broadcast per-decision features to all option rows
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

                # Vectorized per-option features
                _opt_types = np.array([int((_o or {}).get('type', 0) or 0) for _o in _opts], dtype=np.int32)
                _areas = np.array([int((_o or {}).get('area', 0) or 0) for _o in _opts], dtype=np.int32)
                _card_ids = np.array([_v06d7_card_id_from_area(_obs, _o if isinstance(_o, dict) else {}) for _o in _opts], dtype=np.int32)
                _opt_indices = np.array([_v06d7_to_float((_o or {}).get('index'), 0.0) for _o in _opts], dtype=np.float32)
                _in_plays = np.array([_v06d7_to_float((_o or {}).get('inPlayIndex'), 0.0) for _o in _opts], dtype=np.float32)
                _numbers = np.array([_v06d7_to_float((_o or {}).get('number'), 0.0) for _o in _opts], dtype=np.float32)
                _attack_ids = np.array([_v06d7_to_float((_o or {}).get('attackId'), 0.0) for _o in _opts], dtype=np.float32)
                _ability_ids = np.array([_v06d7_to_float((_o or {}).get('abilityId'), 0.0) for _o in _opts], dtype=np.float32)

                _X[_s_row:_e_row, 26] = (_opt_types % 32) / 32.0
                _X[_s_row:_e_row, 27] = (_areas % 16) / 16.0
                _X[_s_row:_e_row, 28] = (_card_ids % 4096) / 4096.0
                _X[_s_row:_e_row, 29] = _opt_indices / 32.0
                _X[_s_row:_e_row, 30] = _in_plays / 16.0
                _X[_s_row:_e_row, 31] = _numbers / 16.0
                _X[_s_row:_e_row, 32] = _attack_ids / 4096.0
                _X[_s_row:_e_row, 33] = _ability_ids / 4096.0

                # One-hot columns via fancy indexing
                _row_idx = np.arange(_s_row, _e_row)
                _X[_row_idx, 34 + (_opt_types % 16)] = 1.0
                _X[_row_idx, 50 + (_areas % 12)] = 1.0
                _X[_row_idx, 62 + (_card_ids % 32)] = 1.0

                # ATTACK / END flags
                _X[_s_row:_e_row, 94] = (_opt_types == 13).astype(np.float32)
                _X[_s_row:_e_row, 95] = (_opt_types == 14).astype(np.float32)

                # Labels
                _y[_s_row + int(_act[0])] = 1.0

                # Option metadata
                for _oi, _opt in enumerate(_opts):
                    _option_meta_rows.append({
                        'decision_id': _decision_id,
                        'option_index': _oi,
                        'label': int(_oi == _act[0]),
                        'option_type': int((_opt or {}).get('type', -1)) if isinstance(_opt, dict) else -1,
                        'option_type_name': _v06d7_name(_V06D7_OPTION_NAMES, int((_opt or {}).get('type', -1)) if isinstance(_opt, dict) else -1, 'OPT'),
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
                    'chosen_option_type_name': _v06d7_name(_V06D7_OPTION_NAMES, int(_chosen_opt.get('type', -1)), 'OPT'),
                    'rule_index': int(_rule_idx),
                    'rule_hit': int(_rule_idx == _act[0]),
                    'rank': _rank,
                    'rank_bucket': _v06d7_rank_bucket(_rank),
                    'won': bool(_won),
                    'matchup': _our_arch + '_vs_' + _opp_arch,
                    'option_start': _s_row,
                    'option_end': _e_row,
                })

                # Save cross-check buffer (first 100 decisions)
                if len(_cross_check_buf) < 100:
                    _cross_check_buf.append((_obs, list(_opts), _nopt, _s_row))

                _cursor += _nopt
                _decision_id += 1

    # Trim to actual rows
    _X = _X[:_cursor]
    _y = _y[:_cursor]

    _t1_extract = _v06d9_time.perf_counter()
    _extract_time_sec = _t1_extract - _t0_extract
    print(f'v0-06d9 feature extraction: {_extract_time_sec:.1f}s, {_cursor} option rows, {_decision_id} decisions')

    _decision_df = pd.DataFrame(_decision_rows)
    _option_meta_df = pd.DataFrame(_option_meta_rows)
    _feature_dim = 96
    if not _cursor or _decision_df.empty:
        raise RuntimeError('no MAIN learning rows were extracted')

    # ------------------------------------------------------------------
    # Cross-check: vectorized vs Python-loop on first 100 decisions
    # ------------------------------------------------------------------
    _check_n = len(_cross_check_buf)
    _check_mismatches = 0
    _check_max_err = 0.0
    for _cc_obs, _cc_opts, _cc_nopt, _cc_start in _cross_check_buf:
        for _cc_oi, _cc_opt in enumerate(_cc_opts):
            _ref = _v06d7_option_features(_cc_obs, _cc_opt if isinstance(_cc_opt, dict) else {}, _cc_oi, _cc_nopt)
            _actual = _X[_cc_start + _cc_oi]
            _errs = [abs(float(a) - float(b)) for a, b in zip(_ref, _actual.tolist())]
            _max_e = max(_errs) if _errs else 0.0
            _check_max_err = max(_check_max_err, _max_e)
            if _max_e > 1e-5:
                _check_mismatches += 1
                if _check_mismatches <= 3:
                    _bad_cols = [i for i, e in enumerate(_errs) if e > 1e-5]
                    print(f'  MISMATCH decision_start={_cc_start} option={_cc_oi} cols={_bad_cols} max_err={_max_e:.2e}')

    _crosscheck_report = {
        'decisions_checked': _check_n,
        'options_checked': sum(t[2] for t in _cross_check_buf),
        'mismatches': _check_mismatches,
        'max_abs_error': float(_check_max_err),
        'passed': _check_mismatches == 0,
    }
    print(f'cross-check: {_crosscheck_report}')
    if _check_mismatches > 0:
        raise RuntimeError(f'feature cross-check FAILED: {_check_mismatches} mismatches, max_err={_check_max_err:.2e}')

    # ------------------------------------------------------------------
    # Save feature matrix as .npy (within-version cache, not for reuse)
    # ------------------------------------------------------------------
    _feat_npy_path = OUTPUT_DIR / (RUN_PREFIX + '-main_feature_matrix.npy')
    _label_npy_path = OUTPUT_DIR / (RUN_PREFIX + '-main_labels.npy')
    np.save(str(_feat_npy_path), _X)
    np.save(str(_label_npy_path), _y)
    ARTIFACT_PATHS['v06d9_main_feature_matrix'] = str(_feat_npy_path)
    ARTIFACT_PATHS['v06d9_main_labels'] = str(_label_npy_path)

    _feature_spec = {
        'feature_dim': _feature_dim,
        'target': 'recorded replay MAIN option index for alakazam actor',
        'label_leakage_guard': 'features are derived from obs/select/option only; recorded action is used only as label',
        'scope': 'SelectContext.MAIN, maxCount=1, alakazam actor',
        'extraction_method': 'numpy_preallocated_vectorized_v06d9',
        'crosscheck_passed': True,
        'crosscheck_max_abs_error': float(_check_max_err),
        'model_adoption': 'fresh offline training plus guarded hybrid evaluation in v0-06d9; no submission torch policy adoption',
        'feature_groups': [
            'selection counts and option position',
            'turn/action flags',
            'player hand/deck/discard/bench/active/prize summaries',
            'option type/area/card id numeric and hashed one-hots',
        ],
    }
    ARTIFACT_PATHS['v06d9_main_feature_spec'] = write_json(OUTPUT_DIR / 'main_feature_spec.json', _feature_spec)

    _dataset_report = {
        'status': 'ok',
        'stats': dict(_stats),
        'decisions': int(len(_decision_df)),
        'option_rows': int(len(_X)),
        'feature_dim': int(_feature_dim),
        'max_decisions': int(_max_decisions),
        'max_options': int(_max_options),
        'split_counts': _decision_df['split'].value_counts().to_dict(),
        'chosen_option_type_counts': _decision_df['chosen_option_type_name'].value_counts().head(30).to_dict(),
        'rule_top1_overall': float(_decision_df['rule_hit'].mean()),
        'rule_exceptions': int(_stats['rule_exceptions']),
        'rule_bad_shape': int(_stats['rule_bad_shape']),
        'extract_time_sec': float(_extract_time_sec),
    }
    ARTIFACT_PATHS['v06d9_main_dataset_report'] = write_json(OUTPUT_DIR / 'main_dataset_report.json', _dataset_report)
    ARTIFACT_PATHS['v06d9_main_decision_rows'] = safe_save_table(_decision_df, OUTPUT_DIR / 'main_decision_rows.parquet')

    _train_ids = set(_decision_df.index[_decision_df['split'].astype(str) == 'train'].tolist())
    _valid_ids = set(_decision_df.index[_decision_df['split'].astype(str) == 'valid'].tolist())
    _holdout_ids = set(_decision_df.index[_decision_df['split'].astype(str) == 'holdout'].tolist())
    if len(_train_ids) < 1000 or len(_holdout_ids) < 200:
        raise RuntimeError('insufficient split counts for MAIN learning')

    _row_decision_ids = np.empty(len(_X), dtype=np.int64)
    for _idx, _r in _decision_df.iterrows():
        _row_decision_ids[int(_r['option_start']):int(_r['option_end'])] = int(_idx)
    _train_mask = np.isin(_row_decision_ids, list(_train_ids))
    _valid_mask = np.isin(_row_decision_ids, list(_valid_ids))
    _holdout_mask = np.isin(_row_decision_ids, list(_holdout_ids))

    # ------------------------------------------------------------------
    # Model definition
    # ------------------------------------------------------------------
    class _V06D7MainScorer(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(dim, 512),
                nn.ReLU(),
                nn.Linear(512, 384),
                nn.ReLU(),
                nn.Linear(384, 256),
                nn.ReLU(),
                nn.Linear(256, 1),
            )

        def forward(self, x):
            return self.net(x).squeeze(-1)

    _device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    _model = _V06D7MainScorer(_feature_dim).to(_device)
    _param_count = int(sum(p.numel() for p in _model.parameters()))
    _epochs = int(os.environ.get('V06D9_MAIN_EPOCHS', os.environ.get('V06D8_MAIN_EPOCHS', '5')))
    _batch = int(os.environ.get('V06D9_MAIN_BATCH_SIZE', os.environ.get('V06D8_MAIN_BATCH_SIZE', '8192')))
    _lr = float(os.environ.get('V06D9_MAIN_LR', os.environ.get('V06D8_MAIN_LR', '0.002')))

    # ------------------------------------------------------------------
    # GPU pre-loading: transfer full train/valid/all tensors once
    # ------------------------------------------------------------------
    _t0_gpu_load = _v06d9_time.perf_counter()
    try:
        _X_train_gpu = torch.tensor(_X[_train_mask], dtype=torch.float32, device=_device)
        _y_train_gpu = torch.tensor(_y[_train_mask], dtype=torch.float32, device=_device)
        _X_valid_gpu = torch.tensor(_X[_valid_mask], dtype=torch.float32, device=_device)
        _y_valid_gpu = torch.tensor(_y[_valid_mask], dtype=torch.float32, device=_device)
        _X_all_gpu = torch.tensor(_X, dtype=torch.float32, device=_device)
        _gpu_preload_ok = True
        print(f'GPU pre-load ok: train={tuple(_X_train_gpu.shape)} valid={tuple(_X_valid_gpu.shape)} all={tuple(_X_all_gpu.shape)} device={_device}')
    except RuntimeError as _gpu_err:
        print(f'GPU pre-load failed ({_gpu_err}), falling back to CPU per-batch transfer')
        _X_train_gpu = torch.tensor(_X[_train_mask], dtype=torch.float32)
        _y_train_gpu = torch.tensor(_y[_train_mask], dtype=torch.float32)
        _X_valid_gpu = torch.tensor(_X[_valid_mask], dtype=torch.float32)
        _y_valid_gpu = torch.tensor(_y[_valid_mask], dtype=torch.float32)
        _X_all_gpu = torch.tensor(_X, dtype=torch.float32)
        _gpu_preload_ok = False
    _t1_gpu_load = _v06d9_time.perf_counter()
    _gpu_load_time_sec = _t1_gpu_load - _t0_gpu_load

    _pos = max(1.0, float(_y[_train_mask].sum()))
    _neg = max(1.0, float(_train_mask.sum() - _y[_train_mask].sum()))
    _loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([min(30.0, _neg / _pos)], dtype=torch.float32, device=_device))
    _opt = torch.optim.AdamW(_model.parameters(), lr=_lr, weight_decay=1e-4)

    # ------------------------------------------------------------------
    # Training loop (no per-batch .to(device) — tensors already on GPU)
    # ------------------------------------------------------------------
    _t0_train = _v06d9_time.perf_counter()
    _history = []
    _n_train = len(_X_train_gpu)
    _n_valid = len(_X_valid_gpu)
    for _ep in range(_epochs):
        _model.train()
        _perm = torch.randperm(_n_train, device=_X_train_gpu.device)
        _losses = []
        for _st in range(0, _n_train, _batch):
            _idx = _perm[_st:_st + _batch]
            _xb = _X_train_gpu[_idx]
            _yb = _y_train_gpu[_idx]
            if _X_train_gpu.device.type != 'cuda':
                _xb = _xb.to(_device)
                _yb = _yb.to(_device)
            _opt.zero_grad(set_to_none=True)
            _loss = _loss_fn(_model(_xb), _yb)
            _loss.backward()
            _opt.step()
            _losses.append(float(_loss.item()))
        _model.eval()
        with torch.no_grad():
            _vl = []
            for _st in range(0, _n_valid, _batch):
                _xb = _X_valid_gpu[_st:_st + _batch]
                _yb = _y_valid_gpu[_st:_st + _batch]
                if _X_valid_gpu.device.type != 'cuda':
                    _xb = _xb.to(_device)
                    _yb = _yb.to(_device)
                _vl.append(float(_loss_fn(_model(_xb), _yb).item()))
        _history.append({'epoch': int(_ep), 'train_loss': float(np.mean(_losses)), 'valid_loss': float(np.mean(_vl))})
        print(f'  epoch {_ep}: train_loss={_history[-1]["train_loss"]:.4f} valid_loss={_history[-1]["valid_loss"]:.4f}')
    _t1_train = _v06d9_time.perf_counter()
    _train_time_sec = _t1_train - _t0_train

    # ------------------------------------------------------------------
    # Inference (GPU batched, no per-batch tensor creation)
    # ------------------------------------------------------------------
    _t0_infer = _v06d9_time.perf_counter()
    _model.eval()
    _all_logits = np.empty(len(_X), dtype=np.float32)
    with torch.no_grad():
        for _st in range(0, len(_X_all_gpu), _batch):
            _xb = _X_all_gpu[_st:_st + _batch]
            if _X_all_gpu.device.type != 'cuda':
                _xb = _xb.to(_device)
            _all_logits[_st:_st + _batch] = _model(_xb).detach().cpu().numpy()
    _t1_infer = _v06d9_time.perf_counter()
    _infer_time_sec = _t1_infer - _t0_infer

    _timing_report = {
        'extract_time_sec': float(_extract_time_sec),
        'gpu_load_time_sec': float(_gpu_load_time_sec),
        'train_time_sec': float(_train_time_sec),
        'infer_time_sec': float(_infer_time_sec),
        'total_pipeline_sec': float(_extract_time_sec + _gpu_load_time_sec + _train_time_sec + _infer_time_sec),
        'gpu_preload_ok': bool(_gpu_preload_ok),
        'device': str(_device),
    }
    print(f'timing: {_timing_report}')

    _model_path = OUTPUT_DIR / (RUN_PREFIX + '-main_option_scorer.pt')
    torch.save(_model.state_dict(), _model_path)
    ARTIFACT_PATHS['v06d9_main_option_scorer'] = str(_model_path)

    # ------------------------------------------------------------------
    # Prediction table and evaluation (identical to v0-06d8)
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
            'random_rank_expected': (int(_r['n_options']) + 1.0) / 2.0,
        })
    _pred_df = pd.DataFrame(_pred_rows)
    ARTIFACT_PATHS['v06d9_main_holdout_predictions'] = safe_save_table(
        _pred_df[_pred_df['split'].astype(str) == 'holdout'],
        OUTPUT_DIR / 'main_holdout_predictions.parquet'
    )

    def _metric_row(df, bucket_name, bucket_value):
        if df.empty:
            return {
                'bucket_name': bucket_name, 'bucket_value': bucket_value, 'n': 0,
                'rule_top1': None, 'model_top1': None, 'model_top3': None,
                'model_chosen_rank': None, 'random_top1_expected': None,
                'model_minus_rule_top1': None,
            }
        return {
            'bucket_name': bucket_name,
            'bucket_value': str(bucket_value),
            'n': int(len(df)),
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
    _hold = _pred_df[_pred_df['split'].astype(str) == 'holdout']
    for _col in ['chosen_option_type_name', 'matchup', 'rank_bucket', 'won']:
        for _val, _g in _hold.groupby(_col, dropna=False):
            _eval_rows.append(_metric_row(_g, _col, _val))
    _eval_df = pd.DataFrame(_eval_rows)
    ARTIFACT_PATHS['v06d9_main_eval_by_bucket'] = safe_save_table(_eval_df, OUTPUT_DIR / 'main_eval_by_bucket.parquet')

    # ------------------------------------------------------------------
    # Hybrid evaluation (identical guard logic to v0-06d8)
    # ------------------------------------------------------------------
    _safe_override_types = [
        x.strip().upper()
        for x in os.environ.get('V06D9_SAFE_MAIN_OVERRIDE_TYPES', os.environ.get('V06D8_SAFE_MAIN_OVERRIDE_TYPES', 'PLAY,ATTACH,EVOLVE')).split(',')
        if x.strip()
    ]
    _rule_veto_types = [
        x.strip().upper()
        for x in os.environ.get('V06D9_RULE_VETO_MAIN_TYPES', os.environ.get('V06D8_RULE_VETO_MAIN_TYPES', 'ATTACK,END,ABILITY')).split(',')
        if x.strip()
    ]
    _prob_grid = [
        float(x) for x in os.environ.get('V06D9_HYBRID_PROB_GRID', os.environ.get('V06D8_HYBRID_PROB_GRID', '0.35,0.45,0.55,0.65,0.75,0.85')).split(',')
        if str(x).strip()
    ]
    _margin_grid = [
        float(x) for x in os.environ.get('V06D9_HYBRID_MARGIN_GRID', os.environ.get('V06D8_HYBRID_MARGIN_GRID', '0.00,0.15,0.30,0.50,0.75,1.00')).split(',')
        if str(x).strip()
    ]

    def _hybrid_eval(df, prob_threshold, margin_threshold, label):
        if df.empty:
            return {
                'label': label, 'n': 0, 'prob_threshold': float(prob_threshold),
                'margin_threshold': float(margin_threshold),
            }
        _d = df.copy()
        _override = (
            _d['model_top1_option_type_name'].astype(str).str.upper().isin(_safe_override_types) &
            (~_d['rule_option_type_name'].astype(str).str.upper().isin(_rule_veto_types)) &
            (_d['model_top1_index'].astype(int) != _d['rule_index'].astype(int)) &
            (_d['model_top1_prob'].fillna(-999.0) >= prob_threshold) &
            (_d['model_margin'].fillna(-999.0) >= margin_threshold)
        )
        _hybrid_hit = np.where(_override, _d['model_top1_hit'].astype(int), _d['rule_hit'].astype(int))
        _benefit = _override & (_d['rule_hit'].astype(int) == 0) & (_d['model_top1_hit'].astype(int) == 1)
        _harm = _override & (_d['rule_hit'].astype(int) == 1) & (_d['model_top1_hit'].astype(int) == 0)
        _neutral = _override & (~_benefit) & (~_harm)
        _danger = _d['chosen_option_type_name'].astype(str).str.upper().isin(['ATTACK', 'END', 'ABILITY'])
        _safe_chosen = _d['chosen_option_type_name'].astype(str).str.upper().isin(_safe_override_types)
        _rule_top1 = float(_d['rule_hit'].mean())
        _model_top1 = float(_d['model_top1_hit'].mean())
        _hybrid_top1 = float(np.mean(_hybrid_hit))
        _danger_rule = float(_d.loc[_danger, 'rule_hit'].mean()) if bool(_danger.any()) else None
        _danger_hybrid = float(np.mean(_hybrid_hit[_danger.to_numpy()])) if bool(_danger.any()) else None
        _safe_rule = float(_d.loc[_safe_chosen, 'rule_hit'].mean()) if bool(_safe_chosen.any()) else None
        _safe_hybrid = float(np.mean(_hybrid_hit[_safe_chosen.to_numpy()])) if bool(_safe_chosen.any()) else None
        return {
            'label': label,
            'n': int(len(_d)),
            'prob_threshold': float(prob_threshold),
            'margin_threshold': float(margin_threshold),
            'safe_override_types': ','.join(_safe_override_types),
            'rule_veto_types': ','.join(_rule_veto_types),
            'rule_top1': _rule_top1,
            'model_top1': _model_top1,
            'hybrid_top1': _hybrid_top1,
            'hybrid_minus_rule_top1': float(_hybrid_top1 - _rule_top1),
            'override_count': int(_override.sum()),
            'override_rate': float(_override.mean()),
            'benefit_count': int(_benefit.sum()),
            'harm_count': int(_harm.sum()),
            'neutral_override_count': int(_neutral.sum()),
            'benefit_rate': float(_benefit.mean()),
            'harm_rate': float(_harm.mean()),
            'benefit_minus_harm_count': int(_benefit.sum() - _harm.sum()),
            'danger_rule_top1': _danger_rule,
            'danger_hybrid_top1': _danger_hybrid,
            'danger_hybrid_minus_rule': None if _danger_rule is None or _danger_hybrid is None else float(_danger_hybrid - _danger_rule),
            'safe_chosen_rule_top1': _safe_rule,
            'safe_chosen_hybrid_top1': _safe_hybrid,
            'safe_chosen_hybrid_minus_rule': None if _safe_rule is None or _safe_hybrid is None else float(_safe_hybrid - _safe_rule),
        }

    _valid_pred = _pred_df[_pred_df['split'].astype(str) == 'valid']
    _threshold_rows = []
    for _pt in _prob_grid:
        for _mt in _margin_grid:
            _row = _hybrid_eval(_valid_pred, _pt, _mt, 'valid_grid')
            _danger_delta = _row.get('danger_hybrid_minus_rule')
            _danger_ok = _danger_delta is None or _danger_delta >= -0.005
            _override_ok = _row.get('override_rate', 0.0) >= 0.02
            _benefit_ok = _row.get('benefit_count', 0) > _row.get('harm_count', 0)
            _row['danger_ok'] = bool(_danger_ok)
            _row['override_ok'] = bool(_override_ok)
            _row['benefit_ok'] = bool(_benefit_ok)
            _row['selection_ok'] = bool(_danger_ok and _override_ok and _benefit_ok)
            _row['selection_score'] = float(
                _row.get('hybrid_minus_rule_top1', 0.0)
                + 0.001 * _row.get('benefit_minus_harm_count', 0)
                - 0.25 * max(0.0, -(_danger_delta or 0.0))
            )
            _threshold_rows.append(_row)
    _threshold_df = pd.DataFrame(_threshold_rows).sort_values(
        ['selection_ok', 'selection_score', 'hybrid_minus_rule_top1', 'override_rate'],
        ascending=[False, False, False, False],
    )
    ARTIFACT_PATHS['v06d9_main_hybrid_threshold_grid'] = safe_save_table(
        _threshold_df,
        OUTPUT_DIR / 'main_hybrid_threshold_grid.parquet'
    )
    _selected_threshold = _threshold_df.iloc[0].to_dict()
    _sel_prob = float(_selected_threshold['prob_threshold'])
    _sel_margin = float(_selected_threshold['margin_threshold'])
    _valid_hybrid_summary = _hybrid_eval(_valid_pred, _sel_prob, _sel_margin, 'valid_selected')
    _holdout_hybrid_summary = _hybrid_eval(_hold, _sel_prob, _sel_margin, 'holdout_selected')

    def _hybrid_bucket_rows(df, prob_threshold, margin_threshold):
        rows = []
        if df.empty:
            return pd.DataFrame()
        for _col in ['chosen_option_type_name', 'matchup', 'rank_bucket', 'won', 'model_top1_option_type_name', 'rule_option_type_name']:
            for _val, _g in df.groupby(_col, dropna=False):
                _r = _hybrid_eval(_g, prob_threshold, margin_threshold, str(_col) + '=' + str(_val))
                _r['bucket_name'] = _col
                _r['bucket_value'] = str(_val)
                rows.append(_r)
        return pd.DataFrame(rows)

    _hybrid_bucket_df = _hybrid_bucket_rows(_hold, _sel_prob, _sel_margin)
    ARTIFACT_PATHS['v06d9_main_hybrid_eval_by_bucket'] = safe_save_table(
        _hybrid_bucket_df,
        OUTPUT_DIR / 'main_hybrid_eval_by_bucket.parquet'
    )

    _hold_summary = _metric_row(_hold, 'split', 'holdout')
    _danger_buckets = _eval_df[
        (_eval_df['bucket_name'] == 'chosen_option_type_name') &
        (_eval_df['bucket_value'].isin(['ATTACK', 'END', 'ABILITY']))
    ].to_dict('records')
    _beats_rule_buckets = _eval_df[
        (_eval_df['bucket_name'] == 'chosen_option_type_name') &
        (_eval_df['n'] >= 100) &
        (_eval_df['model_minus_rule_top1'] > 0.01)
    ].to_dict('records')
    _regresses_danger = [
        r for r in _danger_buckets
        if r.get('n', 0) >= 50 and r.get('model_minus_rule_top1') is not None and r.get('model_minus_rule_top1') < -0.03
    ]

    _hybrid_danger_delta = _holdout_hybrid_summary.get('danger_hybrid_minus_rule')
    _hybrid_harm = int(_holdout_hybrid_summary.get('harm_count', 0) or 0)
    _hybrid_benefit = int(_holdout_hybrid_summary.get('benefit_count', 0) or 0)
    _hybrid_lift = float(_holdout_hybrid_summary.get('hybrid_minus_rule_top1', 0.0) or 0.0)
    _hybrid_override_rate = float(_holdout_hybrid_summary.get('override_rate', 0.0) or 0.0)
    _hybrid_safe_enough = (
        _hybrid_lift > 0.01 and
        _hybrid_override_rate >= 0.02 and
        _hybrid_benefit > _hybrid_harm and
        (_hybrid_danger_delta is None or _hybrid_danger_delta >= -0.005)
    )

    _v06d8_baseline_holdout_model_top1 = 0.5012
    _holdout_model_top1 = float(_hold_summary.get('model_top1', 0.0) or 0.0)
    _quality_ok = abs(_holdout_model_top1 - _v06d8_baseline_holdout_model_top1) <= 0.01

    if _hybrid_safe_enough and _quality_ok:
        _decision = 'needs_followup'
        _reason = 'pipeline speedup confirmed; guarded hybrid improves holdout; runtime probe is next candidate'
    elif not _quality_ok:
        _decision = 'needs_followup'
        _reason = f'pipeline speedup implemented but holdout top-1={_holdout_model_top1:.4f} deviates from v0-06d8 baseline {_v06d8_baseline_holdout_model_top1}; investigate before promoting'
    else:
        _decision = 'needs_followup'
        _reason = 'pipeline speedup confirmed; hybrid not yet safe/useful enough for runtime adoption'

    MAIN_HYBRID_REPORT = {
        'status': 'ok',
        'version': RUN_PREFIX,
        'purpose': 'infrastructure speedup: numpy vectorized feature extraction + gpu pre-loading; same model/features as v0-06d8',
        'crosscheck_report': _crosscheck_report,
        'timing_report': _timing_report,
        'dataset_report': _dataset_report,
        'feature_spec': _feature_spec,
        'model': {
            'param_count': int(_param_count),
            'device': str(_device),
            'epochs': int(_epochs),
            'batch_size': int(_batch),
            'learning_rate': float(_lr),
            'history': _history,
            'model_path': str(_model_path),
        },
        'holdout_summary': _hold_summary,
        'valid_hybrid_summary': _valid_hybrid_summary,
        'holdout_hybrid_summary': _holdout_hybrid_summary,
        'selected_threshold': _selected_threshold,
        'safe_override_types': _safe_override_types,
        'rule_veto_types': _rule_veto_types,
        'dangerous_bucket_summary': _danger_buckets,
        'beats_rule_option_type_buckets': _beats_rule_buckets,
        'v06d8_baseline_holdout_model_top1': _v06d8_baseline_holdout_model_top1,
        'quality_ok': bool(_quality_ok),
        'runtime_adoption': 'disabled_in_v0_06d9',
        'fresh_training': True,
        'reuses_v0_06d8_weights_or_predictions': False,
        'decision': _decision,
        'reason': _reason,
    }
    MAIN_LEARNING_REPORT = MAIN_HYBRID_REPORT
    ARTIFACT_PATHS['v06d9_main_model_report'] = write_json(OUTPUT_DIR / 'main_model_report.json', MAIN_HYBRID_REPORT)
    ARTIFACT_PATHS['v06d9_main_hybrid_report'] = write_json(OUTPUT_DIR / 'main_hybrid_report.json', MAIN_HYBRID_REPORT)

    _promotion = {
        'version': RUN_PREFIX,
        'decision': _decision,
        'promotion_type': 'learning_promote' if _quality_ok else 'needs_followup',
        'reason': _reason,
        'hard_gates': {
            'crosscheck_passed': bool(_crosscheck_report['passed']),
            'crosscheck_max_abs_error': float(_crosscheck_report['max_abs_error']),
            'notebook_stage_status': 'ok',
            'submission_adoption': False,
            'fresh_training': True,
            'reuses_v0_06d8_weights_or_predictions': False,
            'decisions': int(len(_decision_df)),
            'option_rows': int(len(_X)),
            'feature_dim': int(_feature_dim),
            'extract_time_sec': float(_extract_time_sec),
            'train_time_sec': float(_train_time_sec),
            'gpu_preload_ok': bool(_gpu_preload_ok),
            'holdout_rule_top1': _hold_summary.get('rule_top1'),
            'holdout_model_top1': _hold_summary.get('model_top1'),
            'holdout_model_top3': _hold_summary.get('model_top3'),
            'holdout_model_minus_rule_top1': _hold_summary.get('model_minus_rule_top1'),
            'holdout_model_top1_vs_v06d8': float(_holdout_model_top1 - _v06d8_baseline_holdout_model_top1),
            'quality_ok_within_0_01': bool(_quality_ok),
            'selected_prob_threshold': _sel_prob,
            'selected_margin_threshold': _sel_margin,
            'holdout_hybrid_top1': _holdout_hybrid_summary.get('hybrid_top1'),
            'holdout_hybrid_minus_rule_top1': _holdout_hybrid_summary.get('hybrid_minus_rule_top1'),
            'holdout_override_rate': _holdout_hybrid_summary.get('override_rate'),
            'holdout_benefit_count': _holdout_hybrid_summary.get('benefit_count'),
            'holdout_harm_count': _holdout_hybrid_summary.get('harm_count'),
            'holdout_danger_hybrid_minus_rule': _holdout_hybrid_summary.get('danger_hybrid_minus_rule'),
        },
        'known_risks': [
            'MAIN exact imitation is a proxy, not a ladder-strength proof.',
            'Vectorized extraction confirmed correct by cross-check, but only on first 100 decisions.',
            'Runtime adoption is intentionally deferred.',
        ],
        'next_candidates': [
            'v0-06d10: guarded runtime probe — embed MAIN scorer in submission main.py with PLAY/ATTACH/EVOLVE-only overrides.',
            'Add latency/durability evaluation before Kaggle-facing submission.',
        ],
    }
    ARTIFACT_PATHS['v06d9_promotion_decision'] = write_json(OUTPUT_DIR / 'promotion-decision.json', _promotion)

    print('v0-06d9 MAIN hybrid status: ok')
    print(f'  extract={_extract_time_sec:.1f}s  gpu_load={_gpu_load_time_sec:.1f}s  train={_train_time_sec:.1f}s  infer={_infer_time_sec:.1f}s')
    print(f'  crosscheck: passed={_crosscheck_report["passed"]} max_err={_crosscheck_report["max_abs_error"]:.2e}')
    print(f'  holdout model_top1={_holdout_model_top1:.4f} (v06d8 baseline={_v06d8_baseline_holdout_model_top1}, quality_ok={_quality_ok})')
    print(f'  holdout hybrid={_holdout_hybrid_summary.get("hybrid_top1"):.4f} override_rate={_holdout_hybrid_summary.get("override_rate"):.3f}')
    print(f'  decision: {_decision} | {_reason}')

except Exception as _v06d9_exc:
    MAIN_HYBRID_REPORT = {'status': 'error', 'error': repr(_v06d9_exc), 'traceback': traceback.format_exc(limit=8)}
    MAIN_LEARNING_REPORT = MAIN_HYBRID_REPORT
    ARTIFACT_PATHS['v06d9_main_model_report'] = write_json(OUTPUT_DIR / 'main_model_report.json', MAIN_HYBRID_REPORT)
    print('v0-06d9 MAIN hybrid FAILED:', repr(_v06d9_exc))
    raise
