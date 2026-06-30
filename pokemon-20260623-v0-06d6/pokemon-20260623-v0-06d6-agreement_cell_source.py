# =====================================================================
# v0-06d6 replay-agreement diagnostic (outcome/strength/context buckets)
# Diagnostic only: runs the built submission agent on real replay decision
# states and measures agreement with the actual player's recorded action.
# Per-episode hardened (validated at full 5062-episode scale standalone).
# =====================================================================
import importlib.util as _ilu
import math as _math
import tarfile as _tarfile
from collections import defaultdict as _dd

_SELECT_CONTEXT_NAMES = {
    0: 'MAIN', 1: 'SETUP_ACTIVE', 2: 'SETUP_BENCH', 3: 'SWITCH', 4: 'TO_ACTIVE',
    5: 'TO_BENCH', 6: 'TO_FIELD', 7: 'TO_HAND', 8: 'DISCARD', 9: 'TO_DECK',
    10: 'TO_DECK_BOTTOM', 11: 'TO_PRIZE', 12: 'NOT_MOVE', 13: 'DAMAGE_COUNTER',
    14: 'DAMAGE_COUNTER_ANY', 15: 'DAMAGE', 16: 'REMOVE_DAMAGE_COUNTER', 17: 'HEAL',
    18: 'EVOLVES_FROM', 19: 'EVOLVES_TO', 20: 'DEVOLVE', 21: 'ATTACH_FROM',
    22: 'ATTACH_TO', 23: 'DETACH_FROM', 24: 'LOOK', 25: 'EFFECT_TARGET',
    26: 'DISCARD_ENERGY_CARD', 27: 'DISCARD_TOOL_CARD', 28: 'SWITCH_ENERGY_CARD',
    29: 'DISCARD_CARD_OR_ATTACHED_CARD', 30: 'DISCARD_ENERGY', 31: 'TO_HAND_ENERGY',
    32: 'TO_DECK_ENERGY', 33: 'SWITCH_ENERGY', 34: 'SKILL_ORDER', 35: 'ATTACK',
    36: 'DISABLE_ATTACK', 37: 'EVOLVE', 38: 'DRAW_COUNT', 39: 'DAMAGE_COUNTER_COUNT',
    40: 'REMOVE_DAMAGE_COUNTER_COUNT', 41: 'IS_FIRST', 42: 'MULLIGAN', 43: 'ACTIVATE',
    44: 'FIRST_EFFECT', 45: 'MORE_DEVOLVE', 46: 'COIN_HEAD',
    47: 'AFFECT_SPECIAL_CONDITION', 48: 'RECOVER_SPECIAL_CONDITION',
}


def _ctx_name(c):
    try:
        return _SELECT_CONTEXT_NAMES.get(int(c), 'CTX_' + str(c))
    except Exception:
        return 'CTX_' + str(c)


def _archetype_of_deck(deck):
    s = set(int(x) for x in deck)
    if 743 in s:
        return 'alakazam'
    if 878 in s or 879 in s:
        return 'hop_control'
    if 678 in s or 673 in s:
        return 'lucario'
    return 'other'


def _rank_bucket(rank):
    if rank is None:
        return 'unranked'
    if rank <= 10:
        return 'top10'
    if rank <= 50:
        return 'top50'
    if rank <= 200:
        return 'top200'
    return 'other'


def _wilson(w, n, z=1.96):
    if not n:
        return (None, None, None)
    p = w / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    m = z * _math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (p, max(0.0, c - m), min(1.0, c + m))


def _load_agent_from_archive(archive_path):
    with _tarfile.open(archive_path) as t:
        src = t.extractfile('main.py').read().decode()
        deck = [int(x) for x in t.extractfile('deck.csv').read().decode().split() if str(x).strip()]
    mod_path = OUTPUT_DIR / 'pokemon-20260623-v0-06d6-replay_agreement_agent.py'
    if 'write_text' in globals():
        write_text(mod_path, src)
    else:
        open(mod_path, 'w').write(src)
    spec = _ilu.spec_from_file_location('v06d6_agreement_agent', str(mod_path))
    m = _ilu.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m, deck


REPLAY_AGREEMENT_SUMMARY = {'status': 'not_run'}
try:
    from cg.api import to_observation_class as _to_obs  # noqa: F401

    _archive = WORKING_DIR / (RUN_PREFIX + '-submission.tar.gz')
    if not _archive.exists():
        raise FileNotFoundError('submission archive missing: ' + str(_archive))
    _agent_mod, _our_deck = _load_agent_from_archive(_archive)
    _agent = _agent_mod.agent
    _our_arch = _archetype_of_deck(_our_deck)

    _split_map = {}
    if 'EPISODE_SPLIT_DF' in globals() and isinstance(EPISODE_SPLIT_DF, pd.DataFrame) and not EPISODE_SPLIT_DF.empty \
            and 'episode_id' in EPISODE_SPLIT_DF.columns and 'split' in EPISODE_SPLIT_DF.columns:
        for _eid, _sp in zip(EPISODE_SPLIT_DF['episode_id'].astype(str), EPISODE_SPLIT_DF['split'].astype(str)):
            _split_map[_eid] = _sp

    def _rank_of(name):
        info = TOP200_LOOKUP.get(norm_name(name), {}) if 'TOP200_LOOKUP' in globals() else {}
        r = info.get('ranking') if isinstance(info, dict) else None
        return int(r) if r is not None else None

    _stats = _dd(int)
    _by_slice = _dd(lambda: [0, 0])
    _by_rank = _dd(lambda: [0, 0])
    _by_matchup = _dd(lambda: [0, 0])
    _by_split = _dd(lambda: [0, 0])
    _by_ctx = _dd(lambda: [0, 0, 0, 0.0, 0, 0, 0, 0])

    def _process_episode(_d, _fp):
        _eid = str(_d.get('info', {}).get('EpisodeId', Path(_fp).stem))
        _split = _split_map.get(_eid, _split_map.get(Path(_fp).stem, 'unknown'))
        _names = _d.get('info', {}).get('TeamNames') or ['', '']
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
        _seat_arch = {k: _archetype_of_deck(v) for k, v in _seat_deck.items()}

        for _s in _steps:
            if not isinstance(_s, list):
                continue
            for _seat, _a in enumerate(_s):
                if not isinstance(_a, dict) or _a.get('status') != 'ACTIVE':
                    continue
                _obs = _a.get('observation')
                _act = _a.get('action')
                if not isinstance(_obs, dict) or not isinstance(_act, list):
                    continue
                _sel = _obs.get('select')
                if not isinstance(_sel, dict):
                    continue
                _stats['decisions_seen'] += 1
                _opts = _sel.get('option') or []
                _nopt = len(_opts)
                _mx = int(_sel.get('maxCount', 0))
                if _nopt < 2 or not _act or any((not isinstance(x, int) or x < 0 or x >= _nopt) for x in _act) or len(_act) > _mx:
                    _stats['skipped_deck_or_forced'] += 1
                    continue
                _stats['eligible'] += 1
                try:
                    _pred = _agent(_obs)
                except Exception:
                    _stats['agent_exceptions'] += 1
                    continue
                if not isinstance(_pred, list) or any((not isinstance(x, int) or x < 0 or x >= _nopt) for x in _pred):
                    _stats['agent_bad_shape'] += 1
                    continue
                _agree = int(set(_pred) == set(_act)) if _mx > 1 else int(bool(_pred) and _pred[0] == _act[0])
                _actor_arch = _seat_arch.get(_seat, 'other')
                _opp_arch = _seat_arch.get(1 - _seat, 'other')
                _rank = _rank_of(_names[_seat]) if _seat < len(_names) else None
                _rw = _rewards[_seat] if _seat < len(_rewards) else None
                _won = isinstance(_rw, (int, float)) and _rw > 0
                _stats['rank_join_total'] += 1
                if _rank is not None:
                    _stats['rank_join_hits'] += 1
                if _actor_arch == _our_arch:
                    _stats['alakazam_actor_decisions'] += 1
                    _by_slice['all'][0] += _agree; _by_slice['all'][1] += 1
                    if _rank is not None and _rank <= 200:
                        _by_slice['top200_actor'][0] += _agree; _by_slice['top200_actor'][1] += 1
                    if _won:
                        _by_slice['winner_actor'][0] += _agree; _by_slice['winner_actor'][1] += 1
                    else:
                        _by_slice['loser_actor'][0] += _agree; _by_slice['loser_actor'][1] += 1
                    if _rank is not None and _rank <= 200 and _won:
                        _by_slice['top200_winner'][0] += _agree; _by_slice['top200_winner'][1] += 1
                    _rb = _rank_bucket(_rank)
                    _by_rank[_rb][0] += _agree; _by_rank[_rb][1] += 1
                    _mk = _our_arch + '_vs_' + _opp_arch
                    _by_matchup[_mk][0] += _agree; _by_matchup[_mk][1] += 1
                    _by_split[_split][0] += _agree; _by_split[_split][1] += 1
                    _c = _by_ctx[_ctx_name(_sel.get('context'))]
                    _c[0] += _agree; _c[1] += 1; _c[2] += _nopt; _c[3] += 1.0 / max(1, _nopt)
                    if _won:
                        _c[4] += _agree; _c[5] += 1
                    else:
                        _c[6] += _agree; _c[7] += 1

    for _fp in EPISODE_FILES:
        try:
            _d = json.loads(Path(_fp).read_text())
        except Exception:
            _stats['bad_episodes'] += 1
            continue
        if not isinstance(_d, dict):
            _stats['bad_episodes'] += 1
            continue
        try:
            _process_episode(_d, _fp)
            _stats['episodes'] += 1
        except Exception:
            _stats['bad_episodes'] += 1

    def _rows_simple(d, key_name):
        out = []
        for k, (w, n) in d.items():
            p, lo, hi = _wilson(w, n)
            out.append({key_name: k, 'agree': w, 'n': n, 'agreement': p, 'wilson_lo': lo, 'wilson_hi': hi})
        return pd.DataFrame(out).sort_values('n', ascending=False) if out else pd.DataFrame()

    _ctx_rows = []
    for k, v in _by_ctx.items():
        agree, n, sum_nopt, rand_sum, wa, wt, la, lt = v
        ours = agree / max(1, n)
        rand = rand_sum / max(1, n)
        win = (wa / wt) if wt else None
        los = (la / lt) if lt else None
        _ctx_rows.append({
            'context': k, 'n': n, 'agreement': ours, 'random_baseline': rand,
            'lift_vs_random': ours - rand, 'mean_options': sum_nopt / max(1, n),
            'below_random': bool(ours < rand), 'winner_agreement': win, 'loser_agreement': los,
            'winner_minus_loser': (win - los) if (win is not None and los is not None) else None,
            'winner_n': wt, 'loser_n': lt,
        })
    REPLAY_AGREEMENT_BY_CONTEXT_DF = pd.DataFrame(_ctx_rows).sort_values('n', ascending=False) if _ctx_rows else pd.DataFrame()
    REPLAY_AGREEMENT_BY_RANK_DF = _rows_simple(_by_rank, 'rank_bucket')
    REPLAY_AGREEMENT_BY_MATCHUP_DF = _rows_simple(_by_matchup, 'matchup')

    _backlog = []
    for r in _ctx_rows:
        if r['n'] >= 50:
            _backlog.append({**r, 'priority_score': r['n'] * max(0.0, r['random_baseline'] - r['agreement'])})
    REPLAY_DISAGREEMENT_BACKLOG_DF = pd.DataFrame(_backlog).sort_values(
        ['below_random', 'priority_score', 'n'], ascending=[False, False, False]) if _backlog else pd.DataFrame()

    def _slice_dict(d):
        out = {}
        for k, (w, n) in d.items():
            p, lo, hi = _wilson(w, n)
            out[k] = {'agree': w, 'n': n, 'agreement': p, 'wilson_lo': lo, 'wilson_hi': hi}
        return out

    _slc = _slice_dict(_by_slice)
    _winner = _slc.get('winner_actor', {}).get('agreement')
    _loser = _slc.get('loser_actor', {}).get('agreement')
    _rank_vals = [v['agreement'] for v in _slice_dict(_by_rank).values() if v['agreement'] is not None]
    _below_random_ctx = [r['context'] for r in _ctx_rows if r['below_random'] and r['n'] >= 50]
    REPLAY_AGREEMENT_SUMMARY = {
        'status': 'ok',
        'our_archetype': _our_arch,
        'stats': dict(_stats),
        'rank_join_coverage': _stats['rank_join_hits'] / max(1, _stats['rank_join_total']),
        'by_slice': _slc,
        'by_split': _slice_dict(_by_split),
        'discriminating_check': {
            'winner_minus_loser': (_winner - _loser) if (_winner is not None and _loser is not None) else None,
            'rank_bucket_spread': (max(_rank_vals) - min(_rank_vals)) if _rank_vals else None,
            'is_strength_discriminating': bool(
                (_winner is not None and _loser is not None and abs(_winner - _loser) >= 0.03)
                or (len(_rank_vals) >= 2 and (max(_rank_vals) - min(_rank_vals)) >= 0.03)),
            'below_random_contexts': _below_random_ctx,
        },
    }

    ARTIFACT_PATHS['replay_agreement_by_rank'] = safe_save_table(REPLAY_AGREEMENT_BY_RANK_DF, OUTPUT_DIR / 'replay_agreement_by_rank')
    ARTIFACT_PATHS['replay_agreement_by_matchup'] = safe_save_table(REPLAY_AGREEMENT_BY_MATCHUP_DF, OUTPUT_DIR / 'replay_agreement_by_matchup')
    ARTIFACT_PATHS['replay_agreement_by_context'] = safe_save_table(REPLAY_AGREEMENT_BY_CONTEXT_DF, OUTPUT_DIR / 'replay_agreement_by_context')
    ARTIFACT_PATHS['top_disagreement_backlog'] = safe_save_table(REPLAY_DISAGREEMENT_BACKLOG_DF, OUTPUT_DIR / 'top_disagreement_backlog')
    ARTIFACT_PATHS['replay_agreement_summary'] = write_json(OUTPUT_DIR / 'replay_agreement_summary.json', REPLAY_AGREEMENT_SUMMARY)

    print('replay agreement status: ok')
    print('  episodes:', _stats['episodes'], 'bad_episodes:', _stats['bad_episodes'],
          'eligible:', _stats['eligible'], 'alakazam-actor:', _stats['alakazam_actor_decisions'],
          'agent_exceptions:', _stats['agent_exceptions'], 'bad_shape:', _stats['agent_bad_shape'])
    print('  slice all:', _slc.get('all'))
    print('  discriminating:', REPLAY_AGREEMENT_SUMMARY['discriminating_check']['is_strength_discriminating'],
          '| winner-loser:', REPLAY_AGREEMENT_SUMMARY['discriminating_check']['winner_minus_loser'],
          '| below_random_ctx:', _below_random_ctx)
    if not REPLAY_DISAGREEMENT_BACKLOG_DF.empty:
        print(REPLAY_DISAGREEMENT_BACKLOG_DF[['context', 'n', 'agreement', 'random_baseline', 'lift_vs_random', 'below_random']].head(10).to_string(index=False))
except Exception as _agree_exc:
    import traceback as _tb
    REPLAY_AGREEMENT_SUMMARY = {'status': 'error', 'error': repr(_agree_exc), 'traceback': _tb.format_exc()}
    if 'log_error' in globals():
        log_error('replay_agreement_failed', error=repr(_agree_exc))
    print('replay agreement FAILED:', repr(_agree_exc))
