"""Build v09d11 submission = v08d34 agent + 1-ply rollout with LEARNED VALUE eval.

Same winning architecture as v09d4 (1-ply, UNKNOWN_0-only, opponent-agnostic,
roll out rest of my turn with _base_agent) but the end-of-turn eval is the learned
logistic value function P(win|state) instead of the hand-crafted heuristic.
Mirror self-play: learned-eval beats v09d4 (~0.57). Value model embedded inline.
"""
import os, shutil, tarfile, json

SRC_DIR = '/tmp/realagent'
DST_DIR = '/tmp/agent_v09d11'
TAR = '/kaggle/working/pokemon-20260627-v0-09d11-submission-rollout-learnedval.tar.gz'
VM = json.load(open('/kaggle/working/value_model.json'))

with open(os.path.join(SRC_DIR, 'main.py')) as f:
    src = f.read()
assert src.count('def agent(obs_dict') == 1
src = src.replace('def agent(obs_dict', 'def _base_agent(obs_dict')

VMJSON = json.dumps({'feats': VM['feats'], 'mu': VM['mu'], 'sd': VM['sd'],
                     'coef': VM['coef'], 'intercept': VM['intercept']})

ROLLOUT = '''

# ===================== v09d11 ROLLOUT + LEARNED VALUE eval =====================
import ctypes as _ctypes
import json as _json
import math as _math
import cg.api as _rapi
from cg.sim import lib as _rlib

_VM = _json.loads(%r)
_VM_FE, _VM_MU, _VM_SD, _VM_CO, _VM_IC = _VM['feats'], _VM['mu'], _VM['sd'], _VM['coef'], _VM['intercept']


def _rollout_raw_step(search_id, select):
    arr = (_ctypes.c_int * len(select))(*select)
    bs = _rlib.SearchStep(_rapi.agent_ptr, search_id, arr, len(select))
    return _json.loads(bs.decode('utf-8') if isinstance(bs, (bytes, bytearray)) else bs)


def _r_hp(pl):
    t = 0
    for p in (pl.get('active') or []):
        if p: t += int(p.get('hp', 0) or 0)
    for p in (pl.get('bench') or []):
        if p: t += int(p.get('hp', 0) or 0)
    return t


def _r_ids(pl):
    out = []
    for p in (pl.get('active') or []):
        if p: out.append(p.get('id'))
    for p in (pl.get('bench') or []):
        if p: out.append(p.get('id'))
    return out


def _learned_value(obs, p):
    cur = obs.get('current') if obs else None
    if cur is None:
        return -1e9
    me, op = cur['players'][p], cur['players'][1 - p]
    ids = _r_ids(me)
    oa = (op.get('active') or [None]); op_hp = int(oa[0].get('hp', 0)) if (oa and oa[0]) else 0
    ma = (me.get('active') or [None]); my_hp = int(ma[0].get('hp', 0)) if (ma and ma[0]) else 0
    my_act_e = len(ma[0].get('energyCards') or []) if (ma and ma[0]) else 0
    f = {
        'prize_diff': len(op.get('prize') or []) - len(me.get('prize') or []),
        'my_prize': len(me.get('prize') or []), 'op_prize': len(op.get('prize') or []),
        'my_active_hp': my_hp, 'op_active_hp': op_hp,
        'my_total_hp': _r_hp(me), 'op_total_hp': _r_hp(op),
        'my_hand': me.get('handCount', 0) or len(me.get('hand') or []),
        'op_hand': op.get('handCount', 0) or len(op.get('hand') or []),
        'my_deck': me.get('deckCount', 0), 'op_deck': op.get('deckCount', 0),
        'my_bench': len(me.get('bench') or []), 'op_bench': len(op.get('bench') or []),
        'my_active_energy': my_act_e,
        'my_alakazam': sum(1 for i in ids if i == 743),
        'my_kadabra': sum(1 for i in ids if i == 742),
        'my_abra': sum(1 for i in ids if i == 741),
        'turn': cur.get('turn', 0),
    }
    z = _VM_IC
    for i, name in enumerate(_VM_FE):
        z += _VM_CO[i] * ((f.get(name, 0) - _VM_MU[i]) / _VM_SD[i])
    z = max(-30.0, min(30.0, z))
    return 1.0 / (1.0 + _math.exp(-z))


def _rollout_value(child, p, cap=60):
    node = child
    for _ in range(cap):
        st = node.get('state', node)
        obs = st.get('observation')
        sid = st.get('searchId')
        cur = obs.get('current') if obs else None
        if cur is None or cur.get('result', -1) != -1 or cur.get('yourIndex') != p:
            return _learned_value(obs, p)
        if obs.get('select') is None:
            return _learned_value(obs, p)
        try:
            ch = _base_agent(obs)
        except Exception:
            return _learned_value(obs, p)
        if ch is None:
            return _learned_value(obs, p)
        try:
            node = _rollout_raw_step(sid, ch)
            if node.get('error', 0) != 0:
                return _learned_value(obs, p)
        except Exception:
            return _learned_value(obs, p)
    return _learned_value(node.get('state', node).get('observation'), p)


def agent(obs_dict):
    sel = obs_dict.get('select')
    if not sel:
        return _base_agent(obs_dict)
    ctx = sel.get('context')
    n = len(sel.get('option') or [])
    mn = int(sel.get('minCount', 1) or 1)
    if ctx != 0 or n < 2 or mn > 1:
        return _base_agent(obs_dict)
    g = globals()
    snap = (g.get('pre_turn'), g.get('ability_used_dudunsparce'), g.get('ability_used_fezandipiti'))
    try:
        ob = to_observation_class(obs_dict)
        stt = ob.current
        if stt is None or getattr(ob, 'search_begin_input', None) is None:
            return _base_agent(obs_dict)
        p = stt.yourIndex
        me, opp = stt.players[p], stt.players[1 - p]
        yd = list(my_deck); yp = list(my_deck)[:max(1, len(me.prize))]
        od = list(my_deck); op_ = list(my_deck)[:max(1, len(opp.prize))]
        oh = list(my_deck)[:max(1, opp.handCount)]
        oa = [741] if (len(opp.active) > 0 and opp.active[0] is None) else []
        root = _rapi.search_begin(ob, yd, yp, od, op_, oh, oa)
        rid = root.searchId
        best_i, best_v = 0, -1e18
        for oi in range(n):
            try:
                child = _rollout_raw_step(rid, [oi])
                if child.get('error', 0) != 0:
                    continue
                v = _rollout_value(child, p)
            except Exception:
                v = -1e17
            if v > best_v:
                best_v, best_i = v, oi
        try:
            _rapi.search_end()
        except Exception:
            pass
        return [best_i]
    except Exception:
        return _base_agent(obs_dict)
    finally:
        g['pre_turn'], g['ability_used_dudunsparce'], g['ability_used_fezandipiti'] = snap
# =================== end v09d11 ROLLOUT + LEARNED VALUE eval ===================
''' % VMJSON

src = src + ROLLOUT

if os.path.exists(DST_DIR):
    shutil.rmtree(DST_DIR)
os.makedirs(DST_DIR)
shutil.copytree(os.path.join(SRC_DIR, 'cg'), os.path.join(DST_DIR, 'cg'))
for fn in ('unknown0_lgbm.txt', 'unknown0_lgbm_prep.json', 'deck.csv'):
    sp = os.path.join(SRC_DIR, fn)
    if os.path.exists(sp):
        shutil.copy(sp, os.path.join(DST_DIR, fn))
with open(os.path.join(DST_DIR, 'main.py'), 'w') as f:
    f.write(src)

import py_compile
py_compile.compile(os.path.join(DST_DIR, 'main.py'), doraise=True)
print('main.py compiles OK')

shutil.rmtree(os.path.join(DST_DIR, '__pycache__'), ignore_errors=True)
with tarfile.open(TAR, 'w:gz') as tf:
    for fn in ('main.py', 'deck.csv', 'unknown0_lgbm.txt', 'unknown0_lgbm_prep.json'):
        tf.add(os.path.join(DST_DIR, fn), arcname=fn)
    cgd = os.path.join(DST_DIR, 'cg')
    for fn in os.listdir(cgd):
        if 'Zone.Identifier' in fn or fn == '__pycache__':
            continue
        tf.add(os.path.join(cgd, fn), arcname=os.path.join('cg', fn))
print('wrote', TAR)
