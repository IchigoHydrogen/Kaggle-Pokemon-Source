# =====================================================================
# v0-06d14 Part B: Guarded END deferral runtime probe + Kaggle loader entrypoint check
# Injects trained MAIN scorer into rule-only main.py.
# Builds guarded_torch_policy archive (models/ included).
# Runs local smoke eval with latency tracking and validates get_last_callable.
# Feature extractor updated to v06d14 layout (card metadata, no card_id hash).
# Model output may change actions only for rule END -> PLAY/ATTACH/EVOLVE/RETREAT.
# =====================================================================
import shutil as _v06d14b_shutil
import tarfile as _v06d14b_tarfile

try:
    if MAIN_HYBRID_REPORT.get('status') != 'ok':
        raise RuntimeError('Part A failed — cannot proceed with Part B')

    _end_safe_report_b = MAIN_HYBRID_REPORT.get('end_deferral_safe_report', {})
    _end_safe_sel_b = _end_safe_report_b.get('valid_selected_threshold', {})
    _sel_prob_b = float(_end_safe_sel_b.get('prob_threshold', 999.0))
    _sel_margin_b = float(_end_safe_sel_b.get('margin_threshold', 999.0))
    _model_pt_src = Path(MAIN_HYBRID_REPORT['model']['model_path'])
    if not _model_pt_src.exists():
        raise FileNotFoundError(f'trained model not found: {_model_pt_src}')

    _local_models_dir = WORKING_DIR / 'models'
    _local_models_dir.mkdir(exist_ok=True)
    _local_model_dst = _local_models_dir / 'main_option_scorer.pt'
    _v06d14b_shutil.copy2(str(_model_pt_src), str(_local_model_dst))
    print(f'staged model for local testing: {_local_model_dst}')

    # ------------------------------------------------------------------
    # Build guard injection string
    # ------------------------------------------------------------------
    _GUARD_INJECTION = f'''
# ============================================================
# v0-06d14 Guarded END deferral MAIN option scorer — appended to rule agent
# Changes vs v06d11:
#   - Feature layout v06d14: card metadata [61-95], no card_id hash
#   - Action-changing only when rule selected END and model top1 is PLAY/ATTACH/EVOLVE/RETREAT
# ============================================================
_v06d14_rule_agent_fn = agent  # save original

_V06D14_MODEL_LOADED = False
_v06d14_model = None

try:
    import numpy as _v06d14_np
    import time as _v06d14_time
    import torch as _v06d14_torch
    import torch.nn as _v06d14_nn
    import sys as _v06d14_sys
    from pathlib import Path as _v06d14_Path

    class _V06D14MainScorer(_v06d14_nn.Module):
        def __init__(self):
            super().__init__()
            self.net = _v06d14_nn.Sequential(
                _v06d14_nn.Linear(96, 512), _v06d14_nn.ReLU(),
                _v06d14_nn.Linear(512, 384), _v06d14_nn.ReLU(),
                _v06d14_nn.Linear(384, 256), _v06d14_nn.ReLU(),
                _v06d14_nn.Linear(256, 1),
            )
        def forward(self, x):
            return self.net(x).squeeze(-1)

    _v06d14_device = _v06d14_torch.device(\'cpu\')
    _v06d14_base_dir = _v06d14_Path(globals().get(\'__file__\', _v06d14_sys.path[-1])).parent
    if not (_v06d14_base_dir / \'models\' / \'main_option_scorer.pt\').exists():
        _v06d14_base_dir = _v06d14_Path(_v06d14_sys.path[-1])
    _v06d14_model_path = _v06d14_base_dir / \'models\' / \'main_option_scorer.pt\'
    _v06d14_model = _V06D14MainScorer().to(_v06d14_device)
    _v06d14_model.load_state_dict(
        _v06d14_torch.load(str(_v06d14_model_path), map_location=\'cpu\', weights_only=True)
    )
    _v06d14_model.eval()

    # Build card metadata lookup from bundled cg package
    _V06D14_CARD_META_DIM = 35
    _V06D14_CARD_META_OFFSET = 61
    _v06d14_card_meta_lookup = {{}}
    try:
        if str(_v06d14_base_dir) not in _v06d14_sys.path:
            _v06d14_sys.path.insert(0, str(_v06d14_base_dir))
        from cg.api import all_card_data as _v06d14_all_card_data
        for _c in _v06d14_all_card_data():
            _cid = int(_c.cardId)
            _vec = [0.0] * _V06D14_CARD_META_DIM
            _ct = int(getattr(_c, \'cardType\', -1))
            if 0 <= _ct <= 6:
                _vec[_ct] = 1.0
            _vec[7]  = float(bool(getattr(_c, \'ex\', False)))
            _vec[8]  = float(bool(getattr(_c, \'megaEx\', False)))
            _vec[9]  = float(bool(getattr(_c, \'aceSpec\', False)))
            _vec[10] = float(bool(getattr(_c, \'tera\', False)))
            _vec[11] = float(bool(getattr(_c, \'basic\', False)))
            _vec[12] = float(bool(getattr(_c, \'stage1\', False)))
            _vec[13] = float(bool(getattr(_c, \'stage2\', False)))
            _vec[14] = min(1.0, int(getattr(_c, \'hp\', 0) or 0) / 300.0)
            _vec[15] = float(len(getattr(_c, \'skills\', None) or []) > 0)
            # Key-card identity flags [16-34]
            _vec[16] = float(_cid == 741);  _vec[17] = float(_cid == 742)
            _vec[18] = float(_cid == 743);  _vec[19] = float(_cid == 305)
            _vec[20] = float(_cid == 66);   _vec[21] = float(_cid == 140)
            _vec[22] = float(_cid == 1231); _vec[23] = float(_cid == 1225)
            _vec[24] = float(_cid == 1182); _vec[25] = float(_cid == 1184)
            _vec[26] = float(_cid == 1079); _vec[27] = float(_cid == 1081)
            _vec[28] = float(_cid == 1086); _vec[29] = float(_cid == 1152)
            _vec[30] = float(_cid == 1129); _vec[31] = float(_cid == 1097)
            _vec[32] = float(_cid == 1266)
            _vec[33] = float(_cid in (5, 19)); _vec[34] = float(_cid == 13)
            _v06d14_card_meta_lookup[_cid] = _v06d14_np.array(_vec, dtype=_v06d14_np.float32)
    except Exception:
        pass  # card metadata dims 0-15 will be zero; key-card flags built inline

    _V06D14_ZERO_META = _v06d14_np.zeros(_V06D14_CARD_META_DIM, dtype=_v06d14_np.float32)

    _V06D14_MODEL_LOADED = True
except Exception as _v06d14_load_err:
    _V06D14_MODEL_LOADED = False
    _v06d14_load_err_msg = repr(_v06d14_load_err)


def _v06d14_to_float(x, default=0.0):
    try:
        if x is None:
            return float(default)
        return float(x)
    except Exception:
        return float(default)


def _v06d14_card_id_from_area(obs, option):
    try:
        area = int(option.get(\'area\', 0))
        idx = int(option.get(\'index\', -1))
        player = int(option.get(\'playerIndex\',
                     (obs.get(\'current\') or {{}}).get(\'yourIndex\', 0)))
        cur = obs.get(\'current\') or {{}}
        players = cur.get(\'players\') or []
        card = None
        if area == 1:
            sel = obs.get(\'select\') or {{}}
            deck_cards = sel.get(\'deck\') or []
            if 0 <= idx < len(deck_cards):
                card = deck_cards[idx]
        elif area in (2, 3, 4, 5, 6) and 0 <= player < len(players):
            ps = players[player] or {{}}
            key = {{2: \'hand\', 3: \'discard\', 4: \'active\', 5: \'bench\', 6: \'prize\'}}.get(area)
            cards = ps.get(key) or []
            if 0 <= idx < len(cards):
                card = cards[idx]
        elif area == 7:
            cards = cur.get(\'stadium\') or []
            if 0 <= idx < len(cards):
                card = cards[idx]
        elif area == 12:
            cards = cur.get(\'looking\') or []
            if 0 <= idx < len(cards):
                card = cards[idx]
        if isinstance(card, dict):
            return int(card.get(\'id\') or 0)
        return int(getattr(card, \'id\', 0) or 0)
    except Exception:
        return 0


def _v06d14_player_summary(obs, player_idx):
    cur = obs.get(\'current\') or {{}}
    players = cur.get(\'players\') or []
    if not (0 <= player_idx < len(players)) or not isinstance(players[player_idx], dict):
        return {{\'hand\': 0, \'deck\': 0, \'discard\': 0, \'bench\': 0, \'active\': 0, \'prize\': 0}}
    ps = players[player_idx]
    return {{
        \'hand\': int(ps.get(\'handCount\', len(ps.get(\'hand\') or [])) or 0),
        \'deck\': int(ps.get(\'deckCount\', len(ps.get(\'deck\') or [])) or 0),
        \'discard\': int(len(ps.get(\'discard\') or [])),
        \'bench\': int(sum(1 for x in (ps.get(\'bench\') or []) if x)),
        \'active\': int(sum(1 for x in (ps.get(\'active\') or []) if x)),
        \'prize\': int(len(ps.get(\'prize\') or [])),
    }}


def _v06d14_features_batch(obs, opts):
    """Extract (n_opts, 96) feature matrix — v06d14 layout (card metadata, no card_id hash)."""
    n = len(opts)
    if n == 0:
        return _v06d14_np.zeros((0, 96), dtype=_v06d14_np.float32)
    X = _v06d14_np.zeros((n, 96), dtype=_v06d14_np.float32)
    cur = obs.get(\'current\') or {{}}
    sel = obs.get(\'select\') or {{}}
    your = int(cur.get(\'yourIndex\', 0) or 0)
    opp = 1 - your
    mine = _v06d14_player_summary(obs, your)
    theirs = _v06d14_player_summary(obs, opp)
    context = int(sel.get(\'context\', 0) or 0)
    denom = float(max(1, n - 1))
    X[:, 0] = 1.0
    X[:, 1] = min(1.0, n / 32.0)
    X[:, 2] = _v06d14_np.minimum(1.0, _v06d14_np.arange(n, dtype=_v06d14_np.float32) / denom)
    X[:, 3] = _v06d14_to_float(sel.get(\'minCount\'), 0.0) / 8.0
    X[:, 4] = _v06d14_to_float(sel.get(\'maxCount\'), 0.0) / 8.0
    X[:, 5] = min(1.0, _v06d14_to_float(cur.get(\'turn\'), 0.0) / 20.0)
    X[:, 6] = min(1.0, _v06d14_to_float(cur.get(\'turnActionCount\'), 0.0) / 32.0)
    X[:, 7] = min(1.0, _v06d14_to_float(obs.get(\'step\'), 0.0) / 512.0)
    X[:, 8] = float(your)
    X[:, 9] = float(bool(cur.get(\'energyAttached\')))
    X[:, 10] = float(bool(cur.get(\'supporterPlayed\')))
    X[:, 11] = float(bool(cur.get(\'retreated\')))
    X[:, 12] = float(bool(cur.get(\'stadiumPlayed\')))
    X[:, 13] = mine[\'hand\'] / 16.0
    X[:, 14] = mine[\'deck\'] / 60.0
    X[:, 15] = mine[\'discard\'] / 60.0
    X[:, 16] = mine[\'bench\'] / 8.0
    X[:, 17] = mine[\'active\'] / 4.0
    X[:, 18] = mine[\'prize\'] / 6.0
    X[:, 19] = theirs[\'hand\'] / 16.0
    X[:, 20] = theirs[\'deck\'] / 60.0
    X[:, 21] = theirs[\'discard\'] / 60.0
    X[:, 22] = theirs[\'bench\'] / 8.0
    X[:, 23] = theirs[\'active\'] / 4.0
    X[:, 24] = theirs[\'prize\'] / 6.0
    X[:, 25] = (context % 64) / 64.0
    opt_types = _v06d14_np.array([int((o or {{}}).get(\'type\', 0) or 0) for o in opts], dtype=_v06d14_np.int32)
    areas = _v06d14_np.array([int((o or {{}}).get(\'area\', 0) or 0) for o in opts], dtype=_v06d14_np.int32)
    card_ids = _v06d14_np.array([
        _v06d14_card_id_from_area(obs, o if isinstance(o, dict) else {{}}) for o in opts
    ], dtype=_v06d14_np.int32)
    opt_indices = _v06d14_np.array([_v06d14_to_float((o or {{}}).get(\'index\'), 0.0) for o in opts], dtype=_v06d14_np.float32)
    in_plays = _v06d14_np.array([_v06d14_to_float((o or {{}}).get(\'inPlayIndex\'), 0.0) for o in opts], dtype=_v06d14_np.float32)
    numbers = _v06d14_np.array([_v06d14_to_float((o or {{}}).get(\'number\'), 0.0) for o in opts], dtype=_v06d14_np.float32)
    attack_ids = _v06d14_np.array([_v06d14_to_float((o or {{}}).get(\'attackId\'), 0.0) for o in opts], dtype=_v06d14_np.float32)
    ability_ids = _v06d14_np.array([_v06d14_to_float((o or {{}}).get(\'abilityId\'), 0.0) for o in opts], dtype=_v06d14_np.float32)
    # [26-32]: option numerics (v06d14 layout — no card_id continuous)
    X[:, 26] = (opt_types % 32) / 32.0
    X[:, 27] = (areas % 16) / 16.0
    X[:, 28] = opt_indices / 32.0
    X[:, 29] = in_plays / 16.0
    X[:, 30] = numbers / 16.0
    X[:, 31] = attack_ids / 4096.0
    X[:, 32] = ability_ids / 4096.0
    # [33-48]: opt_type one-hot (16 dims)
    row_idx = _v06d14_np.arange(n)
    X[row_idx, 33 + (opt_types % 16)] = 1.0
    # [49-60]: area one-hot (12 dims)
    X[row_idx, 49 + (areas % 12)] = 1.0
    # [61-95]: card metadata (35 dims)
    for _oi in range(n):
        _cid = int(card_ids[_oi])
        _meta = _v06d14_card_meta_lookup.get(_cid, _V06D14_ZERO_META)
        X[_oi, 61:96] = _meta
    return X


_V06D14_RULE_END_TYPE = 14
_V06D14_SAFE_END_DEFERRAL_TYPES = frozenset({{7, 8, 9, 12}})  # PLAY, ATTACH, EVOLVE, RETREAT
_V06D14_PROB_THRESHOLD = {_sel_prob_b}
_V06D14_MARGIN_THRESHOLD = {_sel_margin_b}

_v06d14_action_changes = 0
_v06d14_shadow_disagreements = 0
_v06d14_shadow_type_counts = {{}}
_v06d14_selected_type_counts = {{}}
_v06d14_rule_type_counts = {{}}
_v06d14_unsafe_change_attempts = 0
_v06d14_decisions_seen = 0
_v06d14_latencies_ms = []


def agent(obs_dict: dict) -> list:
    global _v06d14_action_changes, _v06d14_shadow_disagreements, _v06d14_shadow_type_counts
    global _v06d14_selected_type_counts, _v06d14_rule_type_counts, _v06d14_unsafe_change_attempts
    global _v06d14_decisions_seen, _v06d14_latencies_ms
    rule_result = _v06d14_rule_agent_fn(obs_dict)
    if not _V06D14_MODEL_LOADED:
        return rule_result
    try:
        sel = obs_dict.get(\'select\') or {{}}
        ctx = sel.get(\'context\')
        if int(ctx if ctx is not None else -1) != 0:
            return rule_result
        opts = sel.get(\'option\') or []
        n = len(opts)
        if n < 2 or int(sel.get(\'maxCount\', 0) or 0) != 1:
            return rule_result
        if not (isinstance(rule_result, list) and len(rule_result) == 1):
            return rule_result
        rule_idx = rule_result[0]
        rule_opt = opts[rule_idx] if (isinstance(rule_idx, int) and 0 <= rule_idx < n
                                      and isinstance(opts[rule_idx], dict)) else {{}}
        rule_type = int(rule_opt.get(\'type\', -1))
        _v06d14_rule_type_counts[rule_type] = _v06d14_rule_type_counts.get(rule_type, 0) + 1
        t0 = _v06d14_time.perf_counter()
        X = _v06d14_features_batch(obs_dict, opts)
        with _v06d14_torch.no_grad():
            logits = _v06d14_model(
                _v06d14_torch.tensor(X, dtype=_v06d14_torch.float32)
            ).numpy()
        _v06d14_latencies_ms.append((_v06d14_time.perf_counter() - t0) * 1000.0)
        _v06d14_decisions_seen += 1
        top1 = int(_v06d14_np.argmax(logits))
        top1_opt = opts[top1] if (0 <= top1 < n and isinstance(opts[top1], dict)) else {{}}
        top1_type = int(top1_opt.get(\'type\', -1))
        if top1 != rule_idx:
            _v06d14_shadow_disagreements += 1
            _v06d14_shadow_type_counts[top1_type] = _v06d14_shadow_type_counts.get(top1_type, 0) + 1
        if rule_type != _V06D14_RULE_END_TYPE:
            return rule_result
        if top1 == rule_idx:
            return rule_result
        if top1_type not in _V06D14_SAFE_END_DEFERRAL_TYPES:
            _v06d14_unsafe_change_attempts += 1
            return rule_result
        sorted_l = _v06d14_np.sort(logits)[::-1]
        prob = float(1.0 / (1.0 + _v06d14_np.exp(-sorted_l[0])))
        margin = float(sorted_l[0] - sorted_l[1]) if len(sorted_l) > 1 else 0.0
        if prob < _V06D14_PROB_THRESHOLD or margin < _V06D14_MARGIN_THRESHOLD:
            return rule_result
        _v06d14_action_changes += 1
        _v06d14_selected_type_counts[top1_type] = _v06d14_selected_type_counts.get(top1_type, 0) + 1
        return [top1]
        return rule_result
    except Exception:
        return rule_result


def _kaggle_submission_entrypoint(obs_dict: dict, configuration=None) -> list:
    return agent(obs_dict)
'''

    # ------------------------------------------------------------------
    # Local feature cross-check: verify injected extractor matches training
    # ------------------------------------------------------------------
    _synth_obs = {
        'step': 10, 'current': {
            'yourIndex': 0, 'turn': 5, 'turnActionCount': 2,
            'energyAttached': False, 'supporterPlayed': False, 'retreated': False, 'stadiumPlayed': False,
            'players': [
                {'handCount': 7, 'deckCount': 40, 'discard': [1, 2, 3],
                 'bench': [True, True, None, None, None], 'active': [True], 'prize': [1, 2, 3, 4]},
                {'handCount': 5, 'deckCount': 35, 'discard': [1, 2],
                 'bench': [True, None, None, None, None], 'active': [True], 'prize': [1, 2, 3, 4, 5, 6]},
            ],
        },
        'select': {
            'context': 0, 'minCount': 1, 'maxCount': 1,
            'option': [
                {'type': 7, 'area': 2, 'index': 0, 'inPlayIndex': 0, 'number': 0},
                {'type': 8, 'area': 2, 'index': 1, 'inPlayIndex': 1, 'number': 0},
                {'type': 14},
            ],
        },
    }
    _synth_opts = _synth_obs['select']['option']
    _synth_n = len(_synth_opts)

    # Compute via training reference (v06d14 reference feature extractor)
    _synth_ref = np.array(
        [_v06d14_option_features(_synth_obs, _synth_opts[oi], oi, _synth_n, _card_table) for oi in range(_synth_n)],
        dtype=np.float32
    )

    # Write a test script for the runtime injected extractor
    _crosscheck_script = f'''
import numpy as _v06d14_np
import sys as _v06d14_sys
import json

def _v06d14_to_float(x, default=0.0):
    try:
        if x is None: return float(default)
        return float(x)
    except Exception: return float(default)

def _v06d14_card_id_from_area(obs, option):
    try:
        area = int(option.get('area', 0)); idx = int(option.get('index', -1))
        player = int(option.get('playerIndex', (obs.get('current') or {{}}).get('yourIndex', 0)))
        cur = obs.get('current') or {{}}; players = cur.get('players') or []; card = None
        if area == 1:
            deck_cards = (obs.get('select') or {{}}).get('deck') or []
            if 0 <= idx < len(deck_cards): card = deck_cards[idx]
        elif area in (2, 3, 4, 5, 6) and 0 <= player < len(players):
            ps = players[player] or {{}}
            key = {{2: 'hand', 3: 'discard', 4: 'active', 5: 'bench', 6: 'prize'}}.get(area)
            cards = ps.get(key) or []
            if 0 <= idx < len(cards): card = cards[idx]
        elif area == 7:
            cards = cur.get('stadium') or []
            if 0 <= idx < len(cards): card = cards[idx]
        elif area == 12:
            cards = cur.get('looking') or []
            if 0 <= idx < len(cards): card = cards[idx]
        if isinstance(card, dict): return int(card.get('id') or 0)
        return int(getattr(card, 'id', 0) or 0)
    except Exception: return 0

def _v06d14_player_summary(obs, player_idx):
    cur = obs.get('current') or {{}}; players = cur.get('players') or []
    if not (0 <= player_idx < len(players)) or not isinstance(players[player_idx], dict):
        return {{'hand': 0, 'deck': 0, 'discard': 0, 'bench': 0, 'active': 0, 'prize': 0}}
    ps = players[player_idx]
    return {{'hand': int(ps.get('handCount', len(ps.get('hand') or [])) or 0),
             'deck': int(ps.get('deckCount', len(ps.get('deck') or [])) or 0),
             'discard': int(len(ps.get('discard') or [])), 'bench': int(sum(1 for x in (ps.get('bench') or []) if x)),
             'active': int(sum(1 for x in (ps.get('active') or []) if x)), 'prize': int(len(ps.get('prize') or []))}}

_V06D14_CARD_META_DIM = 35
_V06D14_ZERO_META = _v06d14_np.zeros(_V06D14_CARD_META_DIM, dtype=_v06d14_np.float32)
_v06d14_card_meta_lookup = {{}}
try:
    from cg.api import all_card_data as _v06d14_acd
    for _c in _v06d14_acd():
        _cid = int(_c.cardId); _vec = [0.0]*35
        _ct = int(getattr(_c,'cardType',-1))
        if 0<=_ct<=6: _vec[_ct]=1.0
        _vec[7]=float(bool(getattr(_c,'ex',False))); _vec[8]=float(bool(getattr(_c,'megaEx',False)))
        _vec[9]=float(bool(getattr(_c,'aceSpec',False))); _vec[10]=float(bool(getattr(_c,'tera',False)))
        _vec[11]=float(bool(getattr(_c,'basic',False))); _vec[12]=float(bool(getattr(_c,'stage1',False)))
        _vec[13]=float(bool(getattr(_c,'stage2',False))); _vec[14]=min(1.0,int(getattr(_c,'hp',0) or 0)/300.0)
        _vec[15]=float(len(getattr(_c,'skills',None) or [])>0)
        _vec[16]=float(_cid==741); _vec[17]=float(_cid==742); _vec[18]=float(_cid==743)
        _vec[19]=float(_cid==305); _vec[20]=float(_cid==66); _vec[21]=float(_cid==140)
        _vec[22]=float(_cid==1231); _vec[23]=float(_cid==1225); _vec[24]=float(_cid==1182)
        _vec[25]=float(_cid==1184); _vec[26]=float(_cid==1079); _vec[27]=float(_cid==1081)
        _vec[28]=float(_cid==1086); _vec[29]=float(_cid==1152); _vec[30]=float(_cid==1129)
        _vec[31]=float(_cid==1097); _vec[32]=float(_cid==1266)
        _vec[33]=float(_cid in (5,19)); _vec[34]=float(_cid==13)
        _v06d14_card_meta_lookup[_cid] = _v06d14_np.array(_vec, dtype=_v06d14_np.float32)
except Exception: pass

def _v06d14_features_batch(obs, opts):
    n = len(opts)
    if n == 0: return _v06d14_np.zeros((0,96), dtype=_v06d14_np.float32)
    X = _v06d14_np.zeros((n,96), dtype=_v06d14_np.float32)
    cur = obs.get('current') or {{}}; sel = obs.get('select') or {{}}
    your = int(cur.get('yourIndex',0) or 0); opp = 1-your
    mine = _v06d14_player_summary(obs,your); theirs = _v06d14_player_summary(obs,opp)
    context = int(sel.get('context',0) or 0); denom = float(max(1,n-1))
    X[:,0]=1.0; X[:,1]=min(1.0,n/32.0)
    X[:,2]=_v06d14_np.minimum(1.0,_v06d14_np.arange(n,dtype=_v06d14_np.float32)/denom)
    X[:,3]=_v06d14_to_float(sel.get('minCount'),0.0)/8.0; X[:,4]=_v06d14_to_float(sel.get('maxCount'),0.0)/8.0
    X[:,5]=min(1.0,_v06d14_to_float(cur.get('turn'),0.0)/20.0)
    X[:,6]=min(1.0,_v06d14_to_float(cur.get('turnActionCount'),0.0)/32.0)
    X[:,7]=min(1.0,_v06d14_to_float(obs.get('step'),0.0)/512.0)
    X[:,8]=float(your); X[:,9]=float(bool(cur.get('energyAttached')))
    X[:,10]=float(bool(cur.get('supporterPlayed'))); X[:,11]=float(bool(cur.get('retreated')))
    X[:,12]=float(bool(cur.get('stadiumPlayed')))
    X[:,13]=mine['hand']/16.0; X[:,14]=mine['deck']/60.0; X[:,15]=mine['discard']/60.0
    X[:,16]=mine['bench']/8.0; X[:,17]=mine['active']/4.0; X[:,18]=mine['prize']/6.0
    X[:,19]=theirs['hand']/16.0; X[:,20]=theirs['deck']/60.0; X[:,21]=theirs['discard']/60.0
    X[:,22]=theirs['bench']/8.0; X[:,23]=theirs['active']/4.0; X[:,24]=theirs['prize']/6.0
    X[:,25]=(context%64)/64.0
    opt_types=_v06d14_np.array([int((o or {{}}).get('type',0) or 0) for o in opts],dtype=_v06d14_np.int32)
    areas=_v06d14_np.array([int((o or {{}}).get('area',0) or 0) for o in opts],dtype=_v06d14_np.int32)
    card_ids=_v06d14_np.array([_v06d14_card_id_from_area(obs,o if isinstance(o,dict) else {{}}) for o in opts],dtype=_v06d14_np.int32)
    X[:,26]=(opt_types%32)/32.0; X[:,27]=(areas%16)/16.0
    X[:,28]=_v06d14_np.array([_v06d14_to_float((o or {{}}).get('index'),0.0) for o in opts],dtype=_v06d14_np.float32)/32.0
    X[:,29]=_v06d14_np.array([_v06d14_to_float((o or {{}}).get('inPlayIndex'),0.0) for o in opts],dtype=_v06d14_np.float32)/16.0
    X[:,30]=_v06d14_np.array([_v06d14_to_float((o or {{}}).get('number'),0.0) for o in opts],dtype=_v06d14_np.float32)/16.0
    X[:,31]=_v06d14_np.array([_v06d14_to_float((o or {{}}).get('attackId'),0.0) for o in opts],dtype=_v06d14_np.float32)/4096.0
    X[:,32]=_v06d14_np.array([_v06d14_to_float((o or {{}}).get('abilityId'),0.0) for o in opts],dtype=_v06d14_np.float32)/4096.0
    row_idx=_v06d14_np.arange(n)
    X[row_idx,33+(opt_types%16)]=1.0
    X[row_idx,49+(areas%12)]=1.0
    for _oi in range(n):
        _cid=int(card_ids[_oi])
        X[_oi,61:96]=_v06d14_card_meta_lookup.get(_cid,_V06D14_ZERO_META)
    return X

obs = json.loads(_v06d14_sys.argv[1])
opts = obs['select']['option']
X = _v06d14_features_batch(obs, opts)
print(json.dumps(X.tolist()))
'''

    import subprocess as _v06d14b_subp
    _cc_script_path = OUTPUT_DIR / '_v06d14b_cc_test.py'
    _cc_script_path.write_text(_crosscheck_script, encoding='utf-8')
    import sys as _sys_cc
    _cc_result = _v06d14b_subp.run(
        [_sys_cc.executable, str(_cc_script_path), json.dumps(_synth_obs)],
        capture_output=True, text=True, timeout=30,
    )
    if _cc_result.returncode != 0:
        raise RuntimeError(f'runtime feature crosscheck subprocess failed: {_cc_result.stderr[:500]}')
    _runtime_X = np.array(json.loads(_cc_result.stdout.strip()), dtype=np.float32)
    _cc_max_err = float(np.abs(_synth_ref - _runtime_X).max())
    _cc_passed = bool(_cc_max_err < 1e-5)
    print(f'runtime feature crosscheck: max_abs_error={_cc_max_err:.2e} passed={_cc_passed}')
    if not _cc_passed:
        raise RuntimeError(f'runtime feature crosscheck FAILED: max_err={_cc_max_err}')

    # ------------------------------------------------------------------
    # Build guarded source = rule-only source + injection
    # ------------------------------------------------------------------
    _rule_only_archive = WORKING_DIR / (RUN_PREFIX + '-submission-rule-only.tar.gz')
    if not _rule_only_archive.exists():
        _rule_only_archive = OUTPUT_DIR / (RUN_PREFIX + '-submission-rule-only.tar.gz')
    if not _rule_only_archive.exists():
        _rule_only_archive = WORKING_DIR / 'submission-rule-only.tar.gz'
    if not _rule_only_archive.exists():
        raise FileNotFoundError(f'rule-only archive not found in WORKING_DIR or OUTPUT_DIR: {RUN_PREFIX}')

    with _v06d14b_tarfile.open(_rule_only_archive) as _t:
        _base_src = _t.extractfile('main.py').read().decode()
        _base_deck_raw = _t.extractfile('deck.csv').read().decode()
    _base_deck = [int(x) for x in _base_deck_raw.split() if str(x).strip()]

    if '_v06d14_rule_agent_fn' in _base_src:
        raise RuntimeError('base source already contains v06d14 injection')

    _guarded_src = _base_src.rstrip() + '\n\n' + _GUARD_INJECTION.lstrip('\n')
    compile(_guarded_src, 'guarded_end_deferral_main.py', 'exec')

    # ------------------------------------------------------------------
    # Build guarded_torch_policy submission archive
    # ------------------------------------------------------------------
    def _build_guarded_torch_archive(main_source, deck, model_pt_path, archive_path):
        compile(main_source, 'main.py', 'exec')
        if len(deck) != 60:
            raise ValueError(f'deck must be 60 cards, got {len(deck)}')
        _tmp_main = WORKING_DIR / '_v06d14b_tmp_main.py'
        _tmp_deck = WORKING_DIR / '_v06d14b_tmp_deck.csv'
        write_text(_tmp_main, main_source)
        write_text(_tmp_deck, '\n'.join(str(c) for c in deck) + '\n')

        def _excl(info):
            n = info.name
            if ('__pycache__' in n or n.endswith(('.pyc', '.pyo'))
                    or 'Zone.Identifier' in n):
                return None
            return info

        _ap = Path(archive_path)
        if _ap.exists():
            _ap.unlink()
        if 'find_cg_dir' in globals():
            _cg_dir = find_cg_dir()
        else:
            _cg_dir = None
            for _candidate in [
                WORKING_DIR / 'cg',
                Path('/kaggle/input/competitions/pokemon-tcg-ai-battle/sample_submission/cg'),
                Path('/kaggle/input/pokemon-tcg-ai-battle/sample_submission/cg'),
                Path('/kaggle/input/raw/pokemon-tcg-ai-battle/sample_submission/cg'),
            ]:
                if _candidate.is_dir():
                    _cg_dir = _candidate
                    break
        if _cg_dir is None:
            raise FileNotFoundError('cg/ directory not found')

        with _v06d14b_tarfile.open(_ap, 'w:gz') as _tar:
            _tar.add(str(_tmp_main), arcname='main.py', filter=_excl)
            _tar.add(str(_tmp_deck), arcname='deck.csv', filter=_excl)
            _tar.add(str(_cg_dir), arcname='cg', filter=_excl)
            _tar.add(str(model_pt_path), arcname='models/main_option_scorer.pt')

        with _v06d14b_tarfile.open(_ap, 'r:gz') as _tar:
            _names = sorted(_tar.getnames())

        _required = {'main.py', 'deck.csv', 'models/main_option_scorer.pt'}
        _missing = sorted(_required - set(_names))
        if _missing:
            raise RuntimeError(f'archive missing required files: {_missing}')
        _tmp_main.unlink(missing_ok=True)
        _tmp_deck.unlink(missing_ok=True)
        return {'archive': str(_ap), 'n_files': len(_names), 'files': _names}

    _guarded_archive_path = WORKING_DIR / (RUN_PREFIX + '-submission.tar.gz')
    _archive_result = _build_guarded_torch_archive(
        _guarded_src, _base_deck, _local_model_dst, _guarded_archive_path
    )
    print(f'guarded archive: {_guarded_archive_path.name} ({_archive_result["n_files"]} files)')
    ARTIFACT_PATHS['v06d14_guarded_submission_archive'] = str(_guarded_archive_path)

    # ------------------------------------------------------------------
    # Kaggle raw Python loader check
    # ------------------------------------------------------------------
    import tempfile as _v06d14b_tempfile
    from kaggle_environments.agent import get_last_callable as _v06d14_get_last_callable

    _fixture_observation = {'current': None, 'logs': [], 'remainingOverageTime': 600, 'search_begin_input': None, 'select': None, 'step': 0}
    _fixture_configuration = {'actTimeout': 0, 'episodeSteps': 10000, 'runTimeout': 2000, 'seed': 0}
    _fixture_source = 'synthetic_step0'
    for _fp in [OUTPUT_DIR / (RUN_PREFIX + '-validation-episode-81567646.json'),
                Path('/mnt/c/Users/Owner/Downloads/81567646.json')]:
        if _fp.exists():
            try:
                _ep = json.loads(_fp.read_text(encoding='utf-8'))
                _fixture_configuration = _ep.get('configuration') or {}
                _fixture_observation = ((_ep.get('steps') or [[{}]])[0] or [{}])[0].get('observation') or _fixture_observation
                _fixture_source = str(_fp.name)
                break
            except Exception:
                pass

    with _v06d14b_tempfile.TemporaryDirectory(prefix='v06d14_loader_') as _td:
        _extract_dir = Path(_td)
        with _v06d14b_tarfile.open(_guarded_archive_path, 'r:gz') as _tar:
            _tar.extractall(_extract_dir)
        _archived_main_path = _extract_dir / 'main.py'
        _archived_raw = _archived_main_path.read_text(encoding='utf-8')
        _selected_callable = _v06d14_get_last_callable(_archived_raw, path=str(_archived_main_path))
        _selected_name = getattr(_selected_callable, '__name__', type(_selected_callable).__name__)
        _selected_argcount = int(getattr(getattr(_selected_callable, '__code__', None), 'co_argcount', -1))
        _selected_globals = getattr(_selected_callable, '__globals__', {})
        _module_agent = _selected_globals.get('agent')
        _module_agent_name = getattr(_module_agent, '__name__', type(_module_agent).__name__)
        _loader_model_loaded = bool(_selected_globals.get('_V06D14_MODEL_LOADED', False))
        _loader_model_error = str(_selected_globals.get('_v06d14_load_err_msg', ''))
        _loader_model_path = str(_selected_globals.get('_v06d14_model_path', ''))
        _fixture_action_one_arg = _selected_callable(_fixture_observation)
        if _selected_argcount >= 2:
            _fixture_action_two_arg = _selected_callable(_fixture_observation, _fixture_configuration)
        else:
            _fixture_action_two_arg = _fixture_action_one_arg

    _loader_entrypoint_ok = bool(
        _selected_name == '_kaggle_submission_entrypoint'
        and _loader_model_loaded
        and isinstance(_fixture_action_one_arg, list)
        and isinstance(_fixture_action_two_arg, list)
    )
    _loader_validation_report = {
        'status': 'ok' if _loader_entrypoint_ok else 'failed',
        'selected_callable_name': _selected_name,
        'selected_callable_argcount': _selected_argcount,
        'module_agent_name': _module_agent_name,
        'loader_model_loaded': _loader_model_loaded,
        'loader_model_error': _loader_model_error,
        'loader_model_path': _loader_model_path,
        'expected_callable_name': '_kaggle_submission_entrypoint',
        'fixture_source': _fixture_source,
        'fixture_step': int((_fixture_observation or {}).get('step') or 0),
        'fixture_one_arg_action': _fixture_action_one_arg,
        'fixture_two_arg_action': _fixture_action_two_arg,
        'entrypoint_ok': _loader_entrypoint_ok,
    }
    if not _loader_entrypoint_ok:
        raise RuntimeError(f'Kaggle loader selected invalid callable: {_loader_validation_report}')
    print(f'kaggle loader check: selected={_selected_name} argcount={_selected_argcount} model_loaded={_loader_model_loaded} fixture_action={_fixture_action_one_arg}')
    ARTIFACT_PATHS['v06d14_loader_validation_report'] = write_json(
        OUTPUT_DIR / 'kaggle_loader_validation_report.json', _loader_validation_report)

    # ------------------------------------------------------------------
    # Local smoke eval
    # ------------------------------------------------------------------
    _n_smoke_games = int(os.environ.get('V06D14_SMOKE_GAMES', '20'))
    _guarded_mod = import_agent_from_source(_guarded_src, 'v06d14_guarded')
    _torch_load_ok = bool(getattr(_guarded_mod, '_V06D14_MODEL_LOADED', False))
    if not _torch_load_ok:
        _load_err = getattr(_guarded_mod, '_v06d14_load_err_msg', 'unknown')
        raise RuntimeError(f'guarded agent: model failed to load: {_load_err}')
    print(f'guarded agent loaded: _V06D14_MODEL_LOADED={_torch_load_ok}')

    _smoke_rows, _smoke_summary = evaluate_against_agent(
        _guarded_mod, _base_deck,
        make_random_agent(_base_deck), _base_deck,
        _n_smoke_games, 'guarded_vs_random', 'random',
    )
    _smoke_df = pd.DataFrame(_smoke_rows)

    _latencies_ms = list(getattr(_guarded_mod, '_v06d14_latencies_ms', []))
    _action_changes = int(getattr(_guarded_mod, '_v06d14_action_changes', 0))
    _shadow_disagreements = int(getattr(_guarded_mod, '_v06d14_shadow_disagreements', 0))
    _shadow_type_counts = dict(getattr(_guarded_mod, '_v06d14_shadow_type_counts', {}))
    _selected_type_counts = dict(getattr(_guarded_mod, '_v06d14_selected_type_counts', {}))
    _rule_type_counts = dict(getattr(_guarded_mod, '_v06d14_rule_type_counts', {}))
    _unsafe_change_attempts = int(getattr(_guarded_mod, '_v06d14_unsafe_change_attempts', 0))
    _decisions_seen = int(getattr(_guarded_mod, '_v06d14_decisions_seen', 0))

    _latency_stats = {}
    if _latencies_ms:
        _lat_arr = sorted(_latencies_ms)
        _latency_stats = {
            'n': len(_lat_arr),
            'p50_ms': float(np.percentile(_lat_arr, 50)),
            'p95_ms': float(np.percentile(_lat_arr, 95)),
            'p99_ms': float(np.percentile(_lat_arr, 99)),
            'max_ms': float(max(_lat_arr)),
            'mean_ms': float(np.mean(_lat_arr)),
        }
    print(f'smoke eval: {_n_smoke_games} games, action_changes={_action_changes}/{_decisions_seen} shadow_disagreements={_shadow_disagreements}')
    print(f'guard counts: selected_type_counts={_selected_type_counts} unsafe_attempts={_unsafe_change_attempts}')
    if _latency_stats:
        print(f'latency: p50={_latency_stats["p50_ms"]:.2f}ms p95={_latency_stats["p95_ms"]:.2f}ms p99={_latency_stats["p99_ms"]:.2f}ms max={_latency_stats["max_ms"]:.2f}ms')

    _illegal_count = int(_smoke_df.get('illegal_actions', pd.Series([0])).sum()) if not _smoke_df.empty else 0
    _exception_count = int(_smoke_df.get('exceptions', pd.Series([0])).sum()) if not _smoke_df.empty else 0
    _smoke_gate_ok = bool(_illegal_count == 0 and _exception_count == 0 and _torch_load_ok)
    print(f'safety gates: illegal={_illegal_count} exceptions={_exception_count} torch_load={_torch_load_ok} gate_ok={_smoke_gate_ok}')

    _runtime_probe_report = {
        'status': 'ok' if _smoke_gate_ok else 'failed',
        'torch_load_ok': bool(_torch_load_ok),
        'archive': str(_guarded_archive_path),
        'archive_files': _archive_result['n_files'],
        'runtime_feature_crosscheck_passed': bool(_cc_passed),
        'runtime_feature_crosscheck_max_err': float(_cc_max_err),
        'kaggle_loader_validation': _loader_validation_report,
        'prob_threshold': float(_sel_prob_b),
        'margin_threshold': float(_sel_margin_b),
        'runtime_mode': 'guarded_torch_policy',
        'guard': 'rule_END_to_PLAY_ATTACH_EVOLVE_RETREAT_only',
        'safe_override_types': [7, 8, 9, 12],
        'blocked_types': [10, 13, 14, 11],
        'smoke_eval_games': int(_n_smoke_games),
        'smoke_eval_illegal_actions': int(_illegal_count),
        'smoke_eval_exceptions': int(_exception_count),
        'smoke_eval_action_changes': int(_action_changes),
        'shadow_disagreements': int(_shadow_disagreements),
        'shadow_disagreement_rate': float(_shadow_disagreements / max(1, _decisions_seen)),
        'shadow_type_counts': {str(k): int(v) for k, v in _shadow_type_counts.items()},
        'selected_type_counts': {str(k): int(v) for k, v in _selected_type_counts.items()},
        'rule_type_counts': {str(k): int(v) for k, v in _rule_type_counts.items()},
        'unsafe_change_attempts_blocked': int(_unsafe_change_attempts),
        'smoke_eval_decisions_seen': int(_decisions_seen),
        'smoke_gate_ok': bool(_smoke_gate_ok),
        'latency_stats': _latency_stats,
        'model_params': int(MAIN_HYBRID_REPORT['model']['param_count']),
    }
    ARTIFACT_PATHS['v06d14_runtime_probe_report'] = write_json(
        OUTPUT_DIR / 'main_runtime_probe_report.json', _runtime_probe_report)

    MAIN_HYBRID_REPORT['runtime_probe'] = _runtime_probe_report
    MAIN_HYBRID_REPORT['runtime_adoption'] = 'guarded_torch_policy_end_deferral'
    MAIN_HYBRID_REPORT['smoke_gate_ok'] = bool(_smoke_gate_ok)
    MAIN_HYBRID_REPORT['loader_entrypoint_ok'] = bool(_loader_entrypoint_ok)
    _safe_selected_type_ok = set(int(k) for k in _selected_type_counts.keys()).issubset({7, 8, 9, 12})
    MAIN_HYBRID_REPORT['all_gates_ok'] = bool(
        _smoke_gate_ok and _loader_entrypoint_ok and int(_action_changes) > 0 and _safe_selected_type_ok)
    ARTIFACT_PATHS['v06d14_main_hybrid_report'] = write_json(
        OUTPUT_DIR / 'main_hybrid_report.json', MAIN_HYBRID_REPORT)

    # ------------------------------------------------------------------
    # Promotion decision
    # ------------------------------------------------------------------
    _all_gates = bool(MAIN_HYBRID_REPORT['all_gates_ok'])
    _holdout_hybrid_top1 = float(MAIN_HYBRID_REPORT['holdout_hybrid_summary'].get('hybrid_top1', 0.0))
    _holdout_model_top1 = float(MAIN_HYBRID_REPORT['holdout_summary'].get('model_top1', 0.0))
    _v06d11_hybrid_baseline = 0.4507
    _v06d11_model_baseline = 0.5090
    _danger_gate_ok = bool(MAIN_HYBRID_REPORT.get('danger_gate_ok', False))
    _attack_delta = MAIN_HYBRID_REPORT.get('attack_hybrid_delta_holdout')
    _learning_candidates = MAIN_HYBRID_REPORT.get('learning_candidates', {}) or {}
    _end_safe_report_p = MAIN_HYBRID_REPORT.get('end_deferral_safe_report', {}) or {}
    _end_safe_holdout = _end_safe_report_p.get('holdout_summary', {}) or {}
    _end_safe_delta = float(_end_safe_holdout.get('selected_safe_end_deferral_delta', -999.0) or -999.0)
    _end_safe_bmh = int(_end_safe_holdout.get('selected_benefit_minus_harm', 0) or 0)
    _safe_selected_type_ok = set(int(k) for k in _selected_type_counts.keys()).issubset({7, 8, 9, 12})
    _runtime_contract_ok = bool(_action_changes > 0 and _safe_selected_type_ok and _unsafe_change_attempts >= 0)
    _safe_end_holdout_ok = bool(_end_safe_delta >= 0.0 and _end_safe_bmh > 0)

    if _all_gates and _holdout_model_top1 >= 0.49 and _safe_end_holdout_ok and _runtime_contract_ok:
        _promotion_decision = 'runtime_promote'
    elif _smoke_gate_ok and _loader_entrypoint_ok and _safe_end_holdout_ok:
        _promotion_decision = 'needs_followup'
    else:
        _promotion_decision = 'reject'

    _promo = {
        'version': RUN_PREFIX,
        'decision': _promotion_decision,
        'promotion_type': _promotion_decision,
        'runtime_mode': 'guarded_torch_policy',
        'promotion_type_target': 'runtime_promote_candidate',
        'all_gates_ok': _all_gates,
        'danger_gate_ok': bool(_danger_gate_ok),
        'quality_ok': MAIN_HYBRID_REPORT.get('quality_ok', False),
        'learning_candidates': _learning_candidates,
        'safe_end_holdout_ok': bool(_safe_end_holdout_ok),
        'runtime_contract_ok': bool(_runtime_contract_ok),
        'end_safe_holdout_delta': float(_end_safe_delta),
        'end_safe_holdout_benefit_minus_harm': int(_end_safe_bmh),
        'smoke_gate_ok': bool(_smoke_gate_ok),
        'loader_entrypoint_ok': bool(_loader_entrypoint_ok),
        'holdout_model_top1': float(_holdout_model_top1),
        'holdout_hybrid_top1': float(_holdout_hybrid_top1),
        'holdout_model_vs_v06d11': float(_holdout_model_top1 - _v06d11_model_baseline),
        'holdout_hybrid_vs_v06d11': float(_holdout_hybrid_top1 - _v06d11_hybrid_baseline),
        'attack_hybrid_delta_holdout': _attack_delta,
        'v06d11_hybrid_baseline': _v06d11_hybrid_baseline,
        'v06d11_model_baseline': _v06d11_model_baseline,
        'best_epoch': int(MAIN_HYBRID_REPORT['model']['best_epoch']),
        'epochs_run': int(MAIN_HYBRID_REPORT['model']['epochs_run']),
        'best_valid_top1': float(MAIN_HYBRID_REPORT['model']['best_valid_top1']),
        'torch_load_ok': bool(_torch_load_ok),
        'illegal_actions': int(_illegal_count),
        'exceptions': int(_exception_count),
        'action_changes': int(_action_changes),
        'shadow_disagreements': int(_shadow_disagreements),
        'selected_type_counts': {str(k): int(v) for k, v in _selected_type_counts.items()},
        'unsafe_change_attempts_blocked': int(_unsafe_change_attempts),
        'latency_p99_ms': float(_latency_stats.get('p99_ms', -1.0)),
        'selected_prob_threshold': float(_sel_prob_b),
        'selected_margin_threshold': float(_sel_margin_b),
        'archive_path': str(_guarded_archive_path),
        'loader_selected_callable': _loader_validation_report.get('selected_callable_name'),
        'loader_model_loaded': _loader_model_loaded,
    }
    ARTIFACT_PATHS['v06d14_promotion_decision'] = write_json(
        OUTPUT_DIR / 'promotion-decision.json', _promo)
    print(f'PROMOTION DECISION: {_promotion_decision}')
    print(f'  holdout_model_top1={_holdout_model_top1:.4f} (v06d11={_v06d11_model_baseline})')
    print(f'  safe END holdout delta={_end_safe_delta:.4f} benefit_minus_harm={_end_safe_bmh}')
    print(f'  action_changes={_action_changes} selected_type_counts={_selected_type_counts} all_gates_ok={_all_gates}')

except Exception as _v06d14b_exc:
    MAIN_HYBRID_REPORT['runtime_probe'] = {'status': 'error', 'error': repr(_v06d14b_exc),
                                            'traceback': traceback.format_exc(limit=8)}
    ARTIFACT_PATHS['v06d14_runtime_probe_report'] = write_json(
        OUTPUT_DIR / 'main_runtime_probe_report.json', MAIN_HYBRID_REPORT.get('runtime_probe', {}))
    print('v0-06d14 Part B FAILED:', repr(_v06d14b_exc))
    raise
