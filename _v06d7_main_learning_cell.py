# =====================================================================
# v0-06d7 MAIN replay-supervised learning diagnostic
# Diagnostic/offline learning only: do not enable torch in submission here.
# =====================================================================
import importlib.util as _v06d7_ilu
import math as _v06d7_math
import tarfile as _v06d7_tarfile
from collections import defaultdict as _v06d7_dd

MAIN_LEARNING_REPORT = {'status': 'not_run'}

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
    if 'write_text' in globals():
        write_text(mod_path, src)
    else:
        mod_path.write_text(src, encoding='utf-8')
    spec = _v06d7_ilu.spec_from_file_location('v06d7_main_learning_rule_agent', str(mod_path))
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
    row[94] = float(opt_type == 13)  # ATTACK
    row[95] = float(opt_type == 14)  # END
    return row


def _v06d7_logloss_from_logits(logits, labels):
    import torch
    if len(labels) == 0:
        return None
    y = torch.tensor(labels, dtype=torch.float32)
    x = torch.tensor(logits, dtype=torch.float32)
    return float(torch.nn.functional.binary_cross_entropy_with_logits(x, y).item())


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

    _max_decisions = int(os.environ.get('V06D7_MAIN_MAX_DECISIONS', '120000'))
    _max_options = int(os.environ.get('V06D7_MAIN_MAX_OPTIONS', '1600000'))
    _rng = random.Random(20260623)

    _features = []
    _labels = []
    _decision_rows = []
    _option_meta_rows = []
    _decision_id = 0
    _stats = _v06d7_dd(int)

    for _fp in EPISODE_FILES:
        if _decision_id >= _max_decisions or len(_features) >= _max_options:
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
            if _decision_id >= _max_decisions or len(_features) >= _max_options:
                break
            if not isinstance(_s, list):
                continue
            for _seat, _a in enumerate(_s):
                if _decision_id >= _max_decisions or len(_features) >= _max_options:
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
                _start = len(_features)
                for _oi, _opt in enumerate(_opts):
                    _features.append(_v06d7_option_features(_obs, _opt if isinstance(_opt, dict) else {}, _oi, _nopt))
                    _labels.append(1 if _oi == _act[0] else 0)
                    _option_meta_rows.append({
                        'decision_id': _decision_id,
                        'option_index': _oi,
                        'label': int(_oi == _act[0]),
                        'option_type': int((_opt or {}).get('type', -1)) if isinstance(_opt, dict) else -1,
                        'option_type_name': _v06d7_name(_V06D7_OPTION_NAMES, int((_opt or {}).get('type', -1)) if isinstance(_opt, dict) else -1, 'OPT'),
                        'card_id': _v06d7_card_id_from_area(_obs, _opt if isinstance(_opt, dict) else {}),
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
                    'option_start': _start,
                    'option_end': len(_features),
                })
                _decision_id += 1

    _decision_df = pd.DataFrame(_decision_rows)
    _option_meta_df = pd.DataFrame(_option_meta_rows)
    _feature_dim = 96
    if not _features or _decision_df.empty:
        raise RuntimeError('no MAIN learning rows were extracted')
    _X = np.asarray(_features, dtype=np.float32)
    _y = np.asarray(_labels, dtype=np.float32)

    _feature_spec = {
        'feature_dim': _feature_dim,
        'target': 'recorded replay MAIN option index for alakazam actor',
        'label_leakage_guard': 'features are derived from obs/select/option only; recorded action is used only as label',
        'scope': 'SelectContext.MAIN, maxCount=1, alakazam actor',
        'model_adoption': 'offline only in v0-06d7; no submission torch policy adoption',
        'feature_groups': [
            'selection counts and option position',
            'turn/action flags',
            'player hand/deck/discard/bench/active/prize summaries',
            'option type/area/card id numeric and hashed one-hots',
        ],
    }
    ARTIFACT_PATHS['v06d7_main_feature_spec'] = write_json(OUTPUT_DIR / 'main_feature_spec.json', _feature_spec)

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
    }
    ARTIFACT_PATHS['v06d7_main_dataset_report'] = write_json(OUTPUT_DIR / 'main_dataset_report.json', _dataset_report)
    ARTIFACT_PATHS['v06d7_main_decision_rows'] = safe_save_table(_decision_df, OUTPUT_DIR / 'main_decision_rows.parquet')

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
    _epochs = int(os.environ.get('V06D7_MAIN_EPOCHS', '5'))
    _batch = int(os.environ.get('V06D7_MAIN_BATCH_SIZE', '8192'))
    _lr = float(os.environ.get('V06D7_MAIN_LR', '0.002'))
    _pos = max(1.0, float(_y[_train_mask].sum()))
    _neg = max(1.0, float(_train_mask.sum() - _y[_train_mask].sum()))
    _loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([min(30.0, _neg / _pos)], dtype=torch.float32, device=_device))
    _opt = torch.optim.AdamW(_model.parameters(), lr=_lr, weight_decay=1e-4)
    _X_train = torch.tensor(_X[_train_mask], dtype=torch.float32)
    _y_train = torch.tensor(_y[_train_mask], dtype=torch.float32)
    _X_valid = torch.tensor(_X[_valid_mask], dtype=torch.float32)
    _y_valid = torch.tensor(_y[_valid_mask], dtype=torch.float32)
    _history = []
    for _ep in range(_epochs):
        _model.train()
        _perm = torch.randperm(len(_X_train))
        _losses = []
        for _st in range(0, len(_X_train), _batch):
            _idx = _perm[_st:_st + _batch]
            _xb = _X_train[_idx].to(_device)
            _yb = _y_train[_idx].to(_device)
            _opt.zero_grad(set_to_none=True)
            _loss = _loss_fn(_model(_xb), _yb)
            _loss.backward()
            _opt.step()
            _losses.append(float(_loss.item()))
        _model.eval()
        with torch.no_grad():
            _vl = []
            for _st in range(0, len(_X_valid), _batch):
                _xb = _X_valid[_st:_st + _batch].to(_device)
                _yb = _y_valid[_st:_st + _batch].to(_device)
                _vl.append(float(_loss_fn(_model(_xb), _yb).item()))
        _history.append({'epoch': int(_ep), 'train_loss': float(np.mean(_losses)), 'valid_loss': float(np.mean(_vl))})

    _model.eval()
    _all_logits = np.empty(len(_X), dtype=np.float32)
    with torch.no_grad():
        for _st in range(0, len(_X), _batch):
            _xb = torch.tensor(_X[_st:_st + _batch], dtype=torch.float32, device=_device)
            _all_logits[_st:_st + _batch] = _model(_xb).detach().cpu().numpy()

    _model_path = OUTPUT_DIR / (RUN_PREFIX + '-main_option_scorer.pt')
    torch.save(_model.state_dict(), _model_path)
    ARTIFACT_PATHS['v06d7_main_option_scorer'] = str(_model_path)

    _pred_rows = []
    for _idx, _r in _decision_df.iterrows():
        _a = int(_r['option_start'])
        _b = int(_r['option_end'])
        _scores = _all_logits[_a:_b]
        _order = np.argsort(-_scores)
        _chosen_rel = int(_r['chosen_index'])
        _model_top1 = int(_order[0]) if len(_order) else -1
        _rank_pos = int(np.where(_order == _chosen_rel)[0][0] + 1) if _chosen_rel in set(_order.tolist()) else None
        _pred_rows.append({
            'decision_id': int(_idx),
            'split': _r['split'],
            'matchup': _r['matchup'],
            'rank_bucket': _r['rank_bucket'],
            'won': bool(_r['won']),
            'chosen_option_type_name': _r['chosen_option_type_name'],
            'n_options': int(_r['n_options']),
            'rule_hit': int(_r['rule_hit']),
            'model_top1_hit': int(_model_top1 == _chosen_rel),
            'model_top3_hit': int(_chosen_rel in set(_order[:3].tolist())),
            'model_chosen_rank': _rank_pos,
            'random_top1_expected': 1.0 / max(1, int(_r['n_options'])),
            'random_rank_expected': (int(_r['n_options']) + 1.0) / 2.0,
        })
    _pred_df = pd.DataFrame(_pred_rows)
    ARTIFACT_PATHS['v06d7_main_holdout_predictions'] = safe_save_table(
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
    ARTIFACT_PATHS['v06d7_main_eval_by_bucket'] = safe_save_table(_eval_df, OUTPUT_DIR / 'main_eval_by_bucket.parquet')

    _hold_summary = _metric_row(_hold, 'split', 'holdout')
    _danger = _eval_df[
        (_eval_df['bucket_name'] == 'chosen_option_type_name') &
        (_eval_df['bucket_value'].isin(['ATTACK', 'END', 'ABILITY']))
    ].to_dict('records')
    _beats_rule_buckets = _eval_df[
        (_eval_df['bucket_name'] == 'chosen_option_type_name') &
        (_eval_df['n'] >= 100) &
        (_eval_df['model_minus_rule_top1'] > 0.01)
    ].to_dict('records')
    _regresses_danger = [
        r for r in _danger
        if r.get('n', 0) >= 50 and r.get('model_minus_rule_top1') is not None and r.get('model_minus_rule_top1') < -0.03
    ]
    _decision = 'needs_followup'
    _reason = 'offline MAIN learning pipeline executed; submission adoption remains intentionally disabled'
    if _hold_summary['model_minus_rule_top1'] is not None and _hold_summary['model_minus_rule_top1'] <= 0 and not _beats_rule_buckets:
        _decision = 'reject_for_runtime_adoption'
        _reason = 'model did not beat rule baseline on holdout overall or in any sufficiently large option-type bucket'
    elif _regresses_danger:
        _decision = 'needs_followup'
        _reason = 'model has useful buckets but regresses dangerous MAIN option types; runtime adoption blocked'

    MAIN_LEARNING_REPORT = {
        'status': 'ok',
        'version': RUN_PREFIX,
        'purpose': 'MAIN replay-supervised offline PyTorch option scorer; no submission adoption',
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
        'dangerous_bucket_summary': _danger,
        'beats_rule_option_type_buckets': _beats_rule_buckets,
        'runtime_adoption': 'disabled_in_v0_06d7',
        'decision': _decision,
        'reason': _reason,
    }
    ARTIFACT_PATHS['v06d7_main_model_report'] = write_json(OUTPUT_DIR / 'main_model_report.json', MAIN_LEARNING_REPORT)

    _promotion = {
        'version': RUN_PREFIX,
        'decision': _decision,
        'promotion_type': 'offline_main_replay_supervised_learning',
        'reason': _reason,
        'hard_gates': {
            'notebook_stage_status': 'ok',
            'submission_adoption': False,
            'decisions': int(len(_decision_df)),
            'option_rows': int(len(_X)),
            'feature_dim': int(_feature_dim),
            'rule_exceptions': int(_stats['rule_exceptions']),
            'rule_bad_shape': int(_stats['rule_bad_shape']),
            'holdout_rule_top1': _hold_summary.get('rule_top1'),
            'holdout_model_top1': _hold_summary.get('model_top1'),
            'holdout_model_top3': _hold_summary.get('model_top3'),
            'holdout_model_minus_rule_top1': _hold_summary.get('model_minus_rule_top1'),
        },
        'known_risks': [
            'MAIN exact imitation is a proxy, not a ladder-strength proof.',
            'The feature set is still compact and may learn shallow option priors.',
            'Runtime adoption is intentionally deferred until dangerous buckets and guards are designed.',
        ],
        'next_candidates': [
            'Inspect option-type buckets where the offline model beats the rule baseline.',
            'Improve feature extraction if holdout does not beat the rule baseline.',
            'Only later: guarded MAIN torch runtime adoption with hard vetoes for END/ATTACK/ABILITY.',
        ],
    }
    ARTIFACT_PATHS['v06d7_promotion_decision'] = write_json(OUTPUT_DIR / 'promotion-decision.json', _promotion)

    print('v0-06d7 MAIN learning status: ok')
    print('  decisions:', len(_decision_df), 'option_rows:', len(_X), 'param_count:', _param_count)
    print('  holdout:', _hold_summary)
    print('  decision:', _decision, '|', _reason)

except Exception as _v06d7_exc:
    MAIN_LEARNING_REPORT = {'status': 'error', 'error': repr(_v06d7_exc), 'traceback': traceback.format_exc(limit=8)}
    ARTIFACT_PATHS['v06d7_main_model_report'] = write_json(OUTPUT_DIR / 'main_model_report.json', MAIN_LEARNING_REPORT)
    print('v0-06d7 MAIN learning FAILED:', repr(_v06d7_exc))
    raise
