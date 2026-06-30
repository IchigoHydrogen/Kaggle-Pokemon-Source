
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
        player = int(option.get('playerIndex', (obs.get('current') or {}).get('yourIndex', 0)))
        cur = obs.get('current') or {}; players = cur.get('players') or []; card = None
        if area == 1:
            deck_cards = (obs.get('select') or {}).get('deck') or []
            if 0 <= idx < len(deck_cards): card = deck_cards[idx]
        elif area in (2, 3, 4, 5, 6) and 0 <= player < len(players):
            ps = players[player] or {}
            key = {2: 'hand', 3: 'discard', 4: 'active', 5: 'bench', 6: 'prize'}.get(area)
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
    cur = obs.get('current') or {}; players = cur.get('players') or []
    if not (0 <= player_idx < len(players)) or not isinstance(players[player_idx], dict):
        return {'hand': 0, 'deck': 0, 'discard': 0, 'bench': 0, 'active': 0, 'prize': 0}
    ps = players[player_idx]
    return {'hand': int(ps.get('handCount', len(ps.get('hand') or [])) or 0),
             'deck': int(ps.get('deckCount', len(ps.get('deck') or [])) or 0),
             'discard': int(len(ps.get('discard') or [])), 'bench': int(sum(1 for x in (ps.get('bench') or []) if x)),
             'active': int(sum(1 for x in (ps.get('active') or []) if x)), 'prize': int(len(ps.get('prize') or []))}

def _v06d10_features_batch(obs, opts):
    n = len(opts)
    if n == 0: return _v06d10_np.zeros((0, 96), dtype=_v06d10_np.float32)
    X = _v06d10_np.zeros((n, 96), dtype=_v06d10_np.float32)
    cur = obs.get('current') or {}; sel = obs.get('select') or {}
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
    opt_types=_v06d10_np.array([int((o or {}).get('type',0) or 0) for o in opts], dtype=_v06d10_np.int32)
    areas=_v06d10_np.array([int((o or {}).get('area',0) or 0) for o in opts], dtype=_v06d10_np.int32)
    card_ids=_v06d10_np.array([_v06d10_card_id_from_area(obs, o if isinstance(o, dict) else {}) for o in opts], dtype=_v06d10_np.int32)
    opt_indices=_v06d10_np.array([_v06d10_to_float((o or {}).get('index'),0.0) for o in opts], dtype=_v06d10_np.float32)
    in_plays=_v06d10_np.array([_v06d10_to_float((o or {}).get('inPlayIndex'),0.0) for o in opts], dtype=_v06d10_np.float32)
    numbers=_v06d10_np.array([_v06d10_to_float((o or {}).get('number'),0.0) for o in opts], dtype=_v06d10_np.float32)
    attack_ids=_v06d10_np.array([_v06d10_to_float((o or {}).get('attackId'),0.0) for o in opts], dtype=_v06d10_np.float32)
    ability_ids=_v06d10_np.array([_v06d10_to_float((o or {}).get('abilityId'),0.0) for o in opts], dtype=_v06d10_np.float32)
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
