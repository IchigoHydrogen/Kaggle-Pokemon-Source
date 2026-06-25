# =====================================================================
# v0-06d10 Part B: Guarded runtime probe
# Injects trained MAIN scorer into rule-only main.py.
# Builds guarded_torch_policy archive (models/ included).
# Runs local smoke eval with latency tracking.
# =====================================================================
import shutil as _v06d10b_shutil
import tarfile as _v06d10b_tarfile

try:
    if MAIN_HYBRID_REPORT.get('status') != 'ok':
        raise RuntimeError('Part A failed — cannot proceed with Part B')

    _sel_prob_b = float(MAIN_HYBRID_REPORT['selected_prob_threshold'])
    _sel_margin_b = float(MAIN_HYBRID_REPORT['selected_margin_threshold'])
    _model_pt_src = Path(MAIN_HYBRID_REPORT['model']['model_path'])
    if not _model_pt_src.exists():
        raise FileNotFoundError(f'trained model not found: {_model_pt_src}')

    # ------------------------------------------------------------------
    # Stage model for local testing
    # import_agent_from_source writes tmp file to WORKING_DIR;
    # __file__.parent will be WORKING_DIR, so we need models/ there.
    # ------------------------------------------------------------------
    _local_models_dir = WORKING_DIR / 'models'
    _local_models_dir.mkdir(exist_ok=True)
    _local_model_dst = _local_models_dir / 'main_option_scorer.pt'
    _v06d10b_shutil.copy2(str(_model_pt_src), str(_local_model_dst))
    print(f'staged model for local testing: {_local_model_dst}')

    # ------------------------------------------------------------------
    # Build guard injection string (embedded feature extractor + agent)
    # ------------------------------------------------------------------
    _GUARD_INJECTION = f'''
# ============================================================
# v0-06d10 Guarded MAIN option scorer — appended to rule agent
# ============================================================
_v06d10_rule_agent_fn = agent  # save original

_V06D10_MODEL_LOADED = False
_v06d10_model = None

try:
    import numpy as _v06d10_np
    import time as _v06d10_time
    import torch as _v06d10_torch
    import torch.nn as _v06d10_nn
    from pathlib import Path as _v06d10_Path

    class _V06D10MainScorer(_v06d10_nn.Module):
        def __init__(self):
            super().__init__()
            self.net = _v06d10_nn.Sequential(
                _v06d10_nn.Linear(96, 512), _v06d10_nn.ReLU(),
                _v06d10_nn.Linear(512, 384), _v06d10_nn.ReLU(),
                _v06d10_nn.Linear(384, 256), _v06d10_nn.ReLU(),
                _v06d10_nn.Linear(256, 1),
            )
        def forward(self, x):
            return self.net(x).squeeze(-1)

    _v06d10_device = _v06d10_torch.device(\'cpu\')
    _v06d10_model_path = _v06d10_Path(__file__).parent / \'models\' / \'main_option_scorer.pt\'
    _v06d10_model = _V06D10MainScorer().to(_v06d10_device)
    _v06d10_model.load_state_dict(
        _v06d10_torch.load(str(_v06d10_model_path), map_location=\'cpu\', weights_only=True)
    )
    _v06d10_model.eval()
    _V06D10_MODEL_LOADED = True
except Exception as _v06d10_load_err:
    _V06D10_MODEL_LOADED = False
    _v06d10_load_err_msg = repr(_v06d10_load_err)


def _v06d10_to_float(x, default=0.0):
    try:
        if x is None:
            return float(default)
        return float(x)
    except Exception:
        return float(default)


def _v06d10_card_id_from_area(obs, option):
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


def _v06d10_player_summary(obs, player_idx):
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


def _v06d10_features_batch(obs, opts):
    """Extract (n_opts, 96) feature matrix matching training extraction exactly."""
    n = len(opts)
    if n == 0:
        return _v06d10_np.zeros((0, 96), dtype=_v06d10_np.float32)
    X = _v06d10_np.zeros((n, 96), dtype=_v06d10_np.float32)
    cur = obs.get(\'current\') or {{}}
    sel = obs.get(\'select\') or {{}}
    your = int(cur.get(\'yourIndex\', 0) or 0)
    opp = 1 - your
    mine = _v06d10_player_summary(obs, your)
    theirs = _v06d10_player_summary(obs, opp)
    context = int(sel.get(\'context\', 0) or 0)
    denom = float(max(1, n - 1))
    X[:, 0] = 1.0
    X[:, 1] = min(1.0, n / 32.0)
    X[:, 2] = _v06d10_np.minimum(1.0, _v06d10_np.arange(n, dtype=_v06d10_np.float32) / denom)
    X[:, 3] = _v06d10_to_float(sel.get(\'minCount\'), 0.0) / 8.0
    X[:, 4] = _v06d10_to_float(sel.get(\'maxCount\'), 0.0) / 8.0
    X[:, 5] = min(1.0, _v06d10_to_float(cur.get(\'turn\'), 0.0) / 20.0)
    X[:, 6] = min(1.0, _v06d10_to_float(cur.get(\'turnActionCount\'), 0.0) / 32.0)
    X[:, 7] = min(1.0, _v06d10_to_float(obs.get(\'step\'), 0.0) / 512.0)
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
    opt_types = _v06d10_np.array([int((o or {{}}).get(\'type\', 0) or 0) for o in opts], dtype=_v06d10_np.int32)
    areas = _v06d10_np.array([int((o or {{}}).get(\'area\', 0) or 0) for o in opts], dtype=_v06d10_np.int32)
    card_ids = _v06d10_np.array([_v06d10_card_id_from_area(obs, o if isinstance(o, dict) else {{}}) for o in opts], dtype=_v06d10_np.int32)
    opt_indices = _v06d10_np.array([_v06d10_to_float((o or {{}}).get(\'index\'), 0.0) for o in opts], dtype=_v06d10_np.float32)
    in_plays = _v06d10_np.array([_v06d10_to_float((o or {{}}).get(\'inPlayIndex\'), 0.0) for o in opts], dtype=_v06d10_np.float32)
    numbers = _v06d10_np.array([_v06d10_to_float((o or {{}}).get(\'number\'), 0.0) for o in opts], dtype=_v06d10_np.float32)
    attack_ids = _v06d10_np.array([_v06d10_to_float((o or {{}}).get(\'attackId\'), 0.0) for o in opts], dtype=_v06d10_np.float32)
    ability_ids = _v06d10_np.array([_v06d10_to_float((o or {{}}).get(\'abilityId\'), 0.0) for o in opts], dtype=_v06d10_np.float32)
    X[:, 26] = (opt_types % 32) / 32.0
    X[:, 27] = (areas % 16) / 16.0
    X[:, 28] = (card_ids % 4096) / 4096.0
    X[:, 29] = opt_indices / 32.0
    X[:, 30] = in_plays / 16.0
    X[:, 31] = numbers / 16.0
    X[:, 32] = attack_ids / 4096.0
    X[:, 33] = ability_ids / 4096.0
    row_idx = _v06d10_np.arange(n)
    X[row_idx, 34 + (opt_types % 16)] = 1.0
    X[row_idx, 50 + (areas % 12)] = 1.0
    X[row_idx, 62 + (card_ids % 32)] = 1.0
    X[:, 94] = (opt_types == 13).astype(_v06d10_np.float32)
    X[:, 95] = (opt_types == 14).astype(_v06d10_np.float32)
    return X


_V06D10_SAFE_OVERRIDE_TYPES = frozenset({{7, 8, 9}})    # PLAY, ATTACH, EVOLVE
_V06D10_HARD_VETO_TYPES = frozenset({{10, 12, 13, 14}})  # ABILITY, RETREAT, ATTACK, END
_V06D10_PROB_THRESHOLD = {_sel_prob_b}
_V06D10_MARGIN_THRESHOLD = {_sel_margin_b}

_v06d10_action_changes = 0
_v06d10_decisions_seen = 0
_v06d10_latencies_ms = []


def agent(obs_dict: dict) -> list:
    global _v06d10_action_changes, _v06d10_decisions_seen, _v06d10_latencies_ms
    rule_result = _v06d10_rule_agent_fn(obs_dict)
    if not _V06D10_MODEL_LOADED:
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
        if rule_type in _V06D10_HARD_VETO_TYPES:
            return rule_result
        t0 = _v06d10_time.perf_counter()
        X = _v06d10_features_batch(obs_dict, opts)
        with _v06d10_torch.no_grad():
            logits = _v06d10_model(
                _v06d10_torch.tensor(X, dtype=_v06d10_torch.float32)
            ).numpy()
        _v06d10_latencies_ms.append((_v06d10_time.perf_counter() - t0) * 1000.0)
        _v06d10_decisions_seen += 1
        top1 = int(_v06d10_np.argmax(logits))
        top1_opt = opts[top1] if (0 <= top1 < n and isinstance(opts[top1], dict)) else {{}}
        top1_type = int(top1_opt.get(\'type\', -1))
        if top1_type not in _V06D10_SAFE_OVERRIDE_TYPES or top1 == rule_idx:
            return rule_result
        sorted_l = _v06d10_np.sort(logits)[::-1]
        prob = float(1.0 / (1.0 + _v06d10_np.exp(-sorted_l[0])))
        margin = float(sorted_l[0] - sorted_l[1]) if len(sorted_l) > 1 else 0.0
        if prob < _V06D10_PROB_THRESHOLD or margin < _V06D10_MARGIN_THRESHOLD:
            return rule_result
        _v06d10_action_changes += 1
        return [top1]
    except Exception:
        return rule_result
'''

    # ------------------------------------------------------------------
    # Local feature cross-check: 5 synthetic decisions
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

    # Compute via training reference
    _synth_ref = np.array(
        [_v06d7_option_features(_synth_obs, _synth_opts[oi], oi, _synth_n) for oi in range(_synth_n)],
        dtype=np.float32
    )

    # Write a tiny script to test the runtime extractor in isolation
    _crosscheck_script = f'''
import numpy as _v06d10_np
import torch as _v06d10_torch
import torch.nn as _v06d10_nn
import time as _v06d10_time
from pathlib import Path as _v06d10_Path

def _v06d10_to_float(x, default=0.0):
    try:
        if x is None: return float(default)
        return float(x)
    except Exception: return float(default)

def _v06d10_card_id_from_area(obs, option):
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

def _v06d10_player_summary(obs, player_idx):
    cur = obs.get('current') or {{}}; players = cur.get('players') or []
    if not (0 <= player_idx < len(players)) or not isinstance(players[player_idx], dict):
        return {{'hand': 0, 'deck': 0, 'discard': 0, 'bench': 0, 'active': 0, 'prize': 0}}
    ps = players[player_idx]
    return {{'hand': int(ps.get('handCount', len(ps.get('hand') or [])) or 0),
             'deck': int(ps.get('deckCount', len(ps.get('deck') or [])) or 0),
             'discard': int(len(ps.get('discard') or [])), 'bench': int(sum(1 for x in (ps.get('bench') or []) if x)),
             'active': int(sum(1 for x in (ps.get('active') or []) if x)), 'prize': int(len(ps.get('prize') or []))}}

def _v06d10_features_batch(obs, opts):
    n = len(opts)
    if n == 0: return _v06d10_np.zeros((0, 96), dtype=_v06d10_np.float32)
    X = _v06d10_np.zeros((n, 96), dtype=_v06d10_np.float32)
    cur = obs.get('current') or {{}}; sel = obs.get('select') or {{}}
    your = int(cur.get('yourIndex', 0) or 0); opp = 1 - your
    mine = _v06d10_player_summary(obs, your); theirs = _v06d10_player_summary(obs, opp)
    context = int(sel.get('context', 0) or 0); denom = float(max(1, n - 1))
    X[:, 0]=1.0; X[:, 1]=min(1.0, n/32.0)
    X[:, 2]=_v06d10_np.minimum(1.0, _v06d10_np.arange(n, dtype=_v06d10_np.float32)/denom)
    X[:, 3]=_v06d10_to_float(sel.get('minCount'),0.0)/8.0; X[:, 4]=_v06d10_to_float(sel.get('maxCount'),0.0)/8.0
    X[:, 5]=min(1.0, _v06d10_to_float(cur.get('turn'),0.0)/20.0)
    X[:, 6]=min(1.0, _v06d10_to_float(cur.get('turnActionCount'),0.0)/32.0)
    X[:, 7]=min(1.0, _v06d10_to_float(obs.get('step'),0.0)/512.0)
    X[:, 8]=float(your); X[:, 9]=float(bool(cur.get('energyAttached')))
    X[:, 10]=float(bool(cur.get('supporterPlayed'))); X[:, 11]=float(bool(cur.get('retreated')))
    X[:, 12]=float(bool(cur.get('stadiumPlayed')))
    X[:, 13]=mine['hand']/16.0; X[:, 14]=mine['deck']/60.0; X[:, 15]=mine['discard']/60.0
    X[:, 16]=mine['bench']/8.0; X[:, 17]=mine['active']/4.0; X[:, 18]=mine['prize']/6.0
    X[:, 19]=theirs['hand']/16.0; X[:, 20]=theirs['deck']/60.0; X[:, 21]=theirs['discard']/60.0
    X[:, 22]=theirs['bench']/8.0; X[:, 23]=theirs['active']/4.0; X[:, 24]=theirs['prize']/6.0
    X[:, 25]=(context % 64)/64.0
    opt_types=_v06d10_np.array([int((o or {{}}).get('type',0) or 0) for o in opts], dtype=_v06d10_np.int32)
    areas=_v06d10_np.array([int((o or {{}}).get('area',0) or 0) for o in opts], dtype=_v06d10_np.int32)
    card_ids=_v06d10_np.array([_v06d10_card_id_from_area(obs, o if isinstance(o, dict) else {{}}) for o in opts], dtype=_v06d10_np.int32)
    opt_indices=_v06d10_np.array([_v06d10_to_float((o or {{}}).get('index'),0.0) for o in opts], dtype=_v06d10_np.float32)
    in_plays=_v06d10_np.array([_v06d10_to_float((o or {{}}).get('inPlayIndex'),0.0) for o in opts], dtype=_v06d10_np.float32)
    numbers=_v06d10_np.array([_v06d10_to_float((o or {{}}).get('number'),0.0) for o in opts], dtype=_v06d10_np.float32)
    attack_ids=_v06d10_np.array([_v06d10_to_float((o or {{}}).get('attackId'),0.0) for o in opts], dtype=_v06d10_np.float32)
    ability_ids=_v06d10_np.array([_v06d10_to_float((o or {{}}).get('abilityId'),0.0) for o in opts], dtype=_v06d10_np.float32)
    X[:, 26]=(opt_types % 32)/32.0; X[:, 27]=(areas % 16)/16.0; X[:, 28]=(card_ids % 4096)/4096.0
    X[:, 29]=opt_indices/32.0; X[:, 30]=in_plays/16.0; X[:, 31]=numbers/16.0
    X[:, 32]=attack_ids/4096.0; X[:, 33]=ability_ids/4096.0
    row_idx=_v06d10_np.arange(n)
    X[row_idx, 34+(opt_types % 16)]=1.0; X[row_idx, 50+(areas % 12)]=1.0
    X[row_idx, 62+(card_ids % 32)]=1.0
    X[:, 94]=(opt_types == 13).astype(_v06d10_np.float32)
    X[:, 95]=(opt_types == 14).astype(_v06d10_np.float32)
    return X

import json, sys
obs = json.loads(sys.argv[1])
opts = obs['select']['option']
X = _v06d10_features_batch(obs, opts)
print(json.dumps(X.tolist()))
'''

    import subprocess as _v06d10b_subp
    _cc_script_path = OUTPUT_DIR / '_v06d10b_cc_test.py'
    _cc_script_path.write_text(_crosscheck_script, encoding='utf-8')
    _cc_result = _v06d10b_subp.run(
        ['python', str(_cc_script_path), json.dumps(_synth_obs)],
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
        # fall back to the rule-only archive from v0-06d9 naming convention
        _rule_only_archive = WORKING_DIR / 'submission-rule-only.tar.gz'
    if not _rule_only_archive.exists():
        raise FileNotFoundError(f'rule-only archive not found: {_rule_only_archive}')

    with _v06d10b_tarfile.open(_rule_only_archive) as _t:
        _base_src = _t.extractfile('main.py').read().decode()
        _base_deck_raw = _t.extractfile('deck.csv').read().decode()
    _base_deck = [int(x) for x in _base_deck_raw.split() if str(x).strip()]

    # Sanity: base source must NOT already have injection
    if '_v06d10_rule_agent_fn' in _base_src:
        raise RuntimeError('base source already contains v06d10 injection')

    _guarded_src = _base_src.rstrip() + '\n\n' + _GUARD_INJECTION.lstrip('\n')

    # Syntax check
    compile(_guarded_src, 'guarded_main.py', 'exec')
    assert 'import torch' in _guarded_src or '_v06d10_torch' in _guarded_src

    # ------------------------------------------------------------------
    # Build guarded_torch_policy submission archive
    # ------------------------------------------------------------------
    def _build_guarded_torch_archive(main_source, deck, model_pt_path, archive_path):
        compile(main_source, 'main.py', 'exec')
        if len(deck) != 60:
            raise ValueError(f'deck must be 60 cards, got {len(deck)}')

        _tmp_main = WORKING_DIR / '_v06d10b_tmp_main.py'
        _tmp_deck = WORKING_DIR / '_v06d10b_tmp_deck.csv'
        write_text(_tmp_main, main_source)
        write_text(_tmp_deck, '\n'.join(str(c) for c in deck) + '\n')

        def _excl(info):
            n = info.name
            if ('__pycache__' in n or n.endswith(('.pyc', '.pyo'))
                    or n.endswith('.Zone.Identifier') or n.endswith(':Zone.Identifier')
                    or 'Zone.Identifier' in n):
                return None
            return info

        _ap = Path(archive_path)
        if _ap.exists():
            _ap.unlink()

        # Find cg/ directory — use notebook's find_cg_dir() if available
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

        with _v06d10b_tarfile.open(_ap, 'w:gz') as _tar:
            _tar.add(str(_tmp_main), arcname='main.py', filter=_excl)
            _tar.add(str(_tmp_deck), arcname='deck.csv', filter=_excl)
            _tar.add(str(_cg_dir), arcname='cg', filter=_excl)
            _tar.add(str(model_pt_path), arcname='models/main_option_scorer.pt')

        with _v06d10b_tarfile.open(_ap, 'r:gz') as _tar:
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
    ARTIFACT_PATHS['v06d10_guarded_submission_archive'] = str(_guarded_archive_path)

    # ------------------------------------------------------------------
    # Local smoke eval — import guarded agent and run games
    # ------------------------------------------------------------------
    _n_smoke_games = int(os.environ.get('V06D10_SMOKE_GAMES', '20'))
    _guarded_mod = import_agent_from_source(_guarded_src, 'v06d10_guarded')
    _torch_load_ok = bool(getattr(_guarded_mod, '_V06D10_MODEL_LOADED', False))
    if not _torch_load_ok:
        _load_err = getattr(_guarded_mod, '_v06d10_load_err_msg', 'unknown')
        raise RuntimeError(f'guarded agent: model failed to load: {_load_err}')
    print(f'guarded agent loaded: _V06D10_MODEL_LOADED={_torch_load_ok}')

    _smoke_rows, _smoke_summary = evaluate_against_agent(
        _guarded_mod, _our_deck,
        make_random_agent(_our_deck), _our_deck,
        _n_smoke_games, 'guarded_vs_random', 'random',
    )
    _smoke_df = pd.DataFrame(_smoke_rows)

    # Extract per-decision latency and action_changes from guarded module
    _latencies_ms = list(getattr(_guarded_mod, '_v06d10_latencies_ms', []))
    _action_changes = int(getattr(_guarded_mod, '_v06d10_action_changes', 0))
    _decisions_seen = int(getattr(_guarded_mod, '_v06d10_decisions_seen', 0))

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
    print(f'smoke eval: {_n_smoke_games} games, action_changes={_action_changes}/{_decisions_seen}')
    if _latency_stats:
        print(f'latency: p50={_latency_stats["p50_ms"]:.2f}ms p95={_latency_stats["p95_ms"]:.2f}ms p99={_latency_stats["p99_ms"]:.2f}ms max={_latency_stats["max_ms"]:.2f}ms')

    # Safety gates
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
        'prob_threshold': float(_sel_prob_b),
        'margin_threshold': float(_sel_margin_b),
        'safe_override_types': [7, 8, 9],
        'hard_veto_types': [10, 12, 13, 14],
        'smoke_eval_games': int(_n_smoke_games),
        'smoke_eval_illegal_actions': int(_illegal_count),
        'smoke_eval_exceptions': int(_exception_count),
        'smoke_eval_action_changes': int(_action_changes),
        'smoke_eval_decisions_seen': int(_decisions_seen),
        'smoke_gate_ok': bool(_smoke_gate_ok),
        'latency_stats': _latency_stats,
        'model_params': int(MAIN_HYBRID_REPORT['model']['param_count']),
    }
    ARTIFACT_PATHS['v06d10_runtime_probe_report'] = write_json(
        OUTPUT_DIR / 'main_runtime_probe_report.json', _runtime_probe_report)

    # Update MAIN_HYBRID_REPORT with Part B results
    MAIN_HYBRID_REPORT['runtime_probe'] = _runtime_probe_report
    MAIN_HYBRID_REPORT['runtime_adoption'] = 'guarded_torch_policy'
    MAIN_HYBRID_REPORT['smoke_gate_ok'] = bool(_smoke_gate_ok)
    MAIN_HYBRID_REPORT['all_gates_ok'] = bool(MAIN_HYBRID_REPORT.get('quality_ok', False) and _smoke_gate_ok)
    ARTIFACT_PATHS['v06d10_main_hybrid_report'] = write_json(
        OUTPUT_DIR / 'main_hybrid_report.json', MAIN_HYBRID_REPORT)

    # ------------------------------------------------------------------
    # Promotion decision
    # ------------------------------------------------------------------
    _all_gates = bool(MAIN_HYBRID_REPORT['all_gates_ok'])
    _holdout_hybrid_top1 = float(MAIN_HYBRID_REPORT['holdout_hybrid_summary'].get('hybrid_top1', 0.0))
    _v06d9_hybrid_baseline = 0.4454

    if _all_gates:
        if _holdout_hybrid_top1 >= _v06d9_hybrid_baseline - 0.005:
            _promotion_decision = 'runtime_promote'
        else:
            _promotion_decision = 'needs_followup'
    else:
        _promotion_decision = 'reject'

    _promo = {
        'version': RUN_PREFIX,
        'decision': _promotion_decision,
        'all_gates_ok': _all_gates,
        'quality_ok': MAIN_HYBRID_REPORT.get('quality_ok', False),
        'smoke_gate_ok': bool(_smoke_gate_ok),
        'holdout_model_top1': float(MAIN_HYBRID_REPORT['holdout_summary'].get('model_top1', 0.0)),
        'holdout_hybrid_top1': float(_holdout_hybrid_top1),
        'v06d9_hybrid_baseline': _v06d9_hybrid_baseline,
        'holdout_hybrid_minus_v06d9': float(_holdout_hybrid_top1 - _v06d9_hybrid_baseline),
        'best_epoch': int(MAIN_HYBRID_REPORT['model']['best_epoch']),
        'epochs_run': int(MAIN_HYBRID_REPORT['model']['epochs_run']),
        'best_valid_top1': float(MAIN_HYBRID_REPORT['model']['best_valid_top1']),
        'torch_load_ok': bool(_torch_load_ok),
        'illegal_actions': int(_illegal_count),
        'exceptions': int(_exception_count),
        'latency_p99_ms': float(_latency_stats.get('p99_ms', -1.0)),
        'selected_prob_threshold': float(_sel_prob_b),
        'selected_margin_threshold': float(_sel_margin_b),
        'archive_path': str(_guarded_archive_path),
    }
    ARTIFACT_PATHS['v06d10_promotion_decision'] = write_json(
        OUTPUT_DIR / 'promotion-decision.json', _promo)
    print(f'PROMOTION DECISION: {_promotion_decision}')
    print(f'  holdout_hybrid_top1={_holdout_hybrid_top1:.4f} (baseline={_v06d9_hybrid_baseline})')
    print(f'  all_gates_ok={_all_gates}')

except Exception as _v06d10b_exc:
    MAIN_HYBRID_REPORT['runtime_probe'] = {'status': 'error', 'error': repr(_v06d10b_exc),
                                            'traceback': traceback.format_exc(limit=8)}
    ARTIFACT_PATHS['v06d10_runtime_probe_report'] = write_json(
        OUTPUT_DIR / 'main_runtime_probe_report.json', MAIN_HYBRID_REPORT.get('runtime_probe', {}))
    print('v0-06d10 Part B FAILED:', repr(_v06d10b_exc))
    raise
