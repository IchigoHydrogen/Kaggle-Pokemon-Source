
import numpy as _v06d12_np
import sys as _v06d12_sys
import json

def _v06d12_to_float(x, default=0.0):
    try:
        if x is None: return float(default)
        return float(x)
    except Exception: return float(default)

def _v06d12_card_id_from_area(obs, option):
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

def _v06d12_player_summary(obs, player_idx):
    cur = obs.get('current') or {}; players = cur.get('players') or []
    if not (0 <= player_idx < len(players)) or not isinstance(players[player_idx], dict):
        return {'hand': 0, 'deck': 0, 'discard': 0, 'bench': 0, 'active': 0, 'prize': 0}
    ps = players[player_idx]
    return {'hand': int(ps.get('handCount', len(ps.get('hand') or [])) or 0),
             'deck': int(ps.get('deckCount', len(ps.get('deck') or [])) or 0),
             'discard': int(len(ps.get('discard') or [])), 'bench': int(sum(1 for x in (ps.get('bench') or []) if x)),
             'active': int(sum(1 for x in (ps.get('active') or []) if x)), 'prize': int(len(ps.get('prize') or []))}

_V06D12_CARD_META_DIM = 35
_V06D12_ZERO_META = _v06d12_np.zeros(_V06D12_CARD_META_DIM, dtype=_v06d12_np.float32)
_v06d12_card_meta_lookup = {}
try:
    from cg.api import all_card_data as _v06d12_acd
    for _c in _v06d12_acd():
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
        _v06d12_card_meta_lookup[_cid] = _v06d12_np.array(_vec, dtype=_v06d12_np.float32)
except Exception: pass

def _v06d12_features_batch(obs, opts):
    n = len(opts)
    if n == 0: return _v06d12_np.zeros((0,96), dtype=_v06d12_np.float32)
    X = _v06d12_np.zeros((n,96), dtype=_v06d12_np.float32)
    cur = obs.get('current') or {}; sel = obs.get('select') or {}
    your = int(cur.get('yourIndex',0) or 0); opp = 1-your
    mine = _v06d12_player_summary(obs,your); theirs = _v06d12_player_summary(obs,opp)
    context = int(sel.get('context',0) or 0); denom = float(max(1,n-1))
    X[:,0]=1.0; X[:,1]=min(1.0,n/32.0)
    X[:,2]=_v06d12_np.minimum(1.0,_v06d12_np.arange(n,dtype=_v06d12_np.float32)/denom)
    X[:,3]=_v06d12_to_float(sel.get('minCount'),0.0)/8.0; X[:,4]=_v06d12_to_float(sel.get('maxCount'),0.0)/8.0
    X[:,5]=min(1.0,_v06d12_to_float(cur.get('turn'),0.0)/20.0)
    X[:,6]=min(1.0,_v06d12_to_float(cur.get('turnActionCount'),0.0)/32.0)
    X[:,7]=min(1.0,_v06d12_to_float(obs.get('step'),0.0)/512.0)
    X[:,8]=float(your); X[:,9]=float(bool(cur.get('energyAttached')))
    X[:,10]=float(bool(cur.get('supporterPlayed'))); X[:,11]=float(bool(cur.get('retreated')))
    X[:,12]=float(bool(cur.get('stadiumPlayed')))
    X[:,13]=mine['hand']/16.0; X[:,14]=mine['deck']/60.0; X[:,15]=mine['discard']/60.0
    X[:,16]=mine['bench']/8.0; X[:,17]=mine['active']/4.0; X[:,18]=mine['prize']/6.0
    X[:,19]=theirs['hand']/16.0; X[:,20]=theirs['deck']/60.0; X[:,21]=theirs['discard']/60.0
    X[:,22]=theirs['bench']/8.0; X[:,23]=theirs['active']/4.0; X[:,24]=theirs['prize']/6.0
    X[:,25]=(context%64)/64.0
    opt_types=_v06d12_np.array([int((o or {}).get('type',0) or 0) for o in opts],dtype=_v06d12_np.int32)
    areas=_v06d12_np.array([int((o or {}).get('area',0) or 0) for o in opts],dtype=_v06d12_np.int32)
    card_ids=_v06d12_np.array([_v06d12_card_id_from_area(obs,o if isinstance(o,dict) else {}) for o in opts],dtype=_v06d12_np.int32)
    X[:,26]=(opt_types%32)/32.0; X[:,27]=(areas%16)/16.0
    X[:,28]=_v06d12_np.array([_v06d12_to_float((o or {}).get('index'),0.0) for o in opts],dtype=_v06d12_np.float32)/32.0
    X[:,29]=_v06d12_np.array([_v06d12_to_float((o or {}).get('inPlayIndex'),0.0) for o in opts],dtype=_v06d12_np.float32)/16.0
    X[:,30]=_v06d12_np.array([_v06d12_to_float((o or {}).get('number'),0.0) for o in opts],dtype=_v06d12_np.float32)/16.0
    X[:,31]=_v06d12_np.array([_v06d12_to_float((o or {}).get('attackId'),0.0) for o in opts],dtype=_v06d12_np.float32)/4096.0
    X[:,32]=_v06d12_np.array([_v06d12_to_float((o or {}).get('abilityId'),0.0) for o in opts],dtype=_v06d12_np.float32)/4096.0
    row_idx=_v06d12_np.arange(n)
    X[row_idx,33+(opt_types%16)]=1.0
    X[row_idx,49+(areas%12)]=1.0
    for _oi in range(n):
        _cid=int(card_ids[_oi])
        X[_oi,61:96]=_v06d12_card_meta_lookup.get(_cid,_V06D12_ZERO_META)
    return X

obs = json.loads(_v06d12_sys.argv[1])
opts = obs['select']['option']
X = _v06d12_features_batch(obs, opts)
print(json.dumps(X.tolist()))
