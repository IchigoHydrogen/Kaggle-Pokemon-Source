"""v10d2 variant = DUAL-EVAL low-correlation ensemble (both strong, sim-based).

Roll out each ctx0 option once; evaluate the end-of-turn state with TWO different
evals: (1) hand-crafted prize eval (v09d4), (2) the LightGBM learned value (v09d12).
Rank options by each, pick the option minimizing rank_hand + W*rank_gbm. The two
evals are genuinely different views -> lower-correlated errors -> ensemble may beat
either alone (bias-variance-covariance). Ships value_gbm.txt.

Usage: build_v10d2_dual.py <srcdir> <dstdir> <tar> <W>
"""
import os, shutil, tarfile, sys, json

SRC_DIR = sys.argv[1] if len(sys.argv) > 1 else '/tmp/realagent'
DST_DIR = sys.argv[2] if len(sys.argv) > 2 else '/tmp/agent_v10d2_dev2'
TAR = sys.argv[3] if len(sys.argv) > 3 else '/kaggle/working/v10d2_dev2.tar.gz'
W = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0
FEATS = json.load(open('/kaggle/working/value_gbm_feats.json'))['feats']

with open(os.path.join(SRC_DIR, 'main.py')) as f:
    src = f.read()
assert src.count('def agent(obs_dict') == 1
src = src.replace('def agent(obs_dict', 'def _base_agent(obs_dict')

ROLLOUT = ('\n_DUAL_W = ' + repr(W) + '\n_VGBM_FE = ' + repr(FEATS) + '\n') + r'''
# ===================== v10d2 DUAL-EVAL ensemble ===============================
import os as _ros, ctypes as _ctypes, json as _json
import cg.api as _rapi
from cg.sim import lib as _rlib
_VGBM = None
try:
    import lightgbm as _vlgb, numpy as _vnp
    _vp = _ros.path.join(_ros.path.dirname(_ros.path.abspath(__file__)), 'value_gbm.txt')
    if _ros.path.exists(_vp):
        _VGBM = _vlgb.Booster(model_file=_vp)
except Exception:
    _VGBM = None


def _d_raw_step(sid, sel):
    arr = (_ctypes.c_int * len(sel))(*sel)
    bs = _rlib.SearchStep(_rapi.agent_ptr, sid, arr, len(sel))
    return _json.loads(bs.decode('utf-8') if isinstance(bs, (bytes, bytearray)) else bs)


def _d_hp(pl):
    t = 0
    for p in (pl.get('active') or []):
        if p: t += int(p.get('hp', 0) or 0)
    for p in (pl.get('bench') or []):
        if p: t += int(p.get('hp', 0) or 0)
    return t


def _d_ids(pl):
    out = []
    for p in (pl.get('active') or []):
        if p: out.append(p.get('id'))
    for p in (pl.get('bench') or []):
        if p: out.append(p.get('id'))
    return out


def _hand_eval(obs, p):
    cur = obs.get('current') if obs else None
    if cur is None: return -1e9
    me, op = cur['players'][p], cur['players'][1 - p]
    mp = len(me.get('prize') or []); opp = len(op.get('prize') or [])
    oa = (op.get('active') or [None]); op_hp = int(oa[0].get('hp', 0)) if (oa and oa[0]) else 0
    mh = me.get('handCount', 0) or len(me.get('hand') or [])
    return 1000.0 * (opp - mp) - 2.0 * op_hp + (_d_hp(me) - _d_hp(op)) + 5.0 * mh


def _gbm_eval(obs, p):
    # v10d2 dev-eval: causal tempo/development (NO learned model -> no confound)
    cur = obs.get('current') if obs else None
    if cur is None: return -1e9
    me = cur['players'][p]
    ids = _d_ids(me)
    ma = (me.get('active') or [None]); my_e = len(ma[0].get('energyCards') or []) if (ma and ma[0]) else 0
    my_act_id = ma[0].get('id') if (ma and ma[0]) else 0
    alak_inplay = 1 if (my_act_id == 743 or 743 in ids) else 0
    my_bench = len(me.get('bench') or [])
    my_deck = me.get('deckCount', 0)
    my_hand = me.get('handCount', 0) or len(me.get('hand') or [])
    kad = sum(1 for i in ids if i == 742); abra = sum(1 for i in ids if i == 741)
    op = cur['players'][1 - p]
    op_bench = len(op.get('bench') or [])
    attack_ready = 1 if (alak_inplay and my_e >= 2) else 0
    return (40.0 * alak_inplay + 8.0 * my_e + 4.0 * my_bench + 3.0 * my_hand
            - 60.0 * (1 if my_deck <= 3 else 0)
            + 25.0 * attack_ready + 5.0 * kad + 2.0 * abra - 3.0 * op_bench)


def _d_rollout_end(child, p, cap=60):
    node = child
    for _ in range(cap):
        st = node.get('state', node); obs = st.get('observation'); sid = st.get('searchId')
        cur = obs.get('current') if obs else None
        if cur is None or cur.get('result', -1) != -1 or cur.get('yourIndex') != p:
            return obs
        if obs.get('select') is None: return obs
        try: ch = _base_agent(obs)
        except Exception: return obs
        if ch is None: return obs
        try:
            node = _d_raw_step(sid, ch)
            if node.get('error', 0) != 0: return obs
        except Exception: return obs
    return node.get('state', node).get('observation')


def _ranks(vals):
    order = sorted(range(len(vals)), key=lambda i: -vals[i])
    r = [0] * len(vals)
    for rank, i in enumerate(order):
        r[i] = rank
    return r


def agent(obs_dict):
    sel = obs_dict.get('select')
    if not sel: return _base_agent(obs_dict)
    ctx = sel.get('context'); n = len(sel.get('option') or []); mn = int(sel.get('minCount', 1) or 1)
    if ctx != 0 or n < 2 or mn > 1:
        return _base_agent(obs_dict)
    g = globals()
    snap = (g.get('pre_turn'), g.get('ability_used_dudunsparce'), g.get('ability_used_fezandipiti'))
    try:
        ob = to_observation_class(obs_dict); stt = ob.current
        if stt is None or getattr(ob, 'search_begin_input', None) is None:
            return _base_agent(obs_dict)
        p = stt.yourIndex; me, opp = stt.players[p], stt.players[1 - p]
        yd = list(my_deck); yp = list(my_deck)[:max(1, len(me.prize))]
        od = list(my_deck); op_ = list(my_deck)[:max(1, len(opp.prize))]
        oh = list(my_deck)[:max(1, opp.handCount)]
        oa = [741] if (len(opp.active) > 0 and opp.active[0] is None) else []
        root = _rapi.search_begin(ob, yd, yp, od, op_, oh, oa); rid = root.searchId
        hv, gv = [], []
        for oi in range(n):
            try:
                child = _d_raw_step(rid, [oi])
                end = _d_rollout_end(child, p) if child.get('error', 0) == 0 else None
            except Exception:
                end = None
            hv.append(_hand_eval(end, p) if end else -1e17)
            gv.append(_gbm_eval(end, p) if end else -1e17)
        try: _rapi.search_end()
        except Exception: pass
        rh, rg = _ranks(hv), _ranks(gv)
        combined = [rh[i] + _DUAL_W * rg[i] for i in range(n)]
        return [int(min(range(n), key=lambda i: combined[i]))]
    except Exception:
        return _base_agent(obs_dict)
    finally:
        g['pre_turn'], g['ability_used_dudunsparce'], g['ability_used_fezandipiti'] = snap
# =================== end v10d2 DUAL-EVAL ensemble =============================
'''

src = src + ROLLOUT
if os.path.exists(DST_DIR):
    shutil.rmtree(DST_DIR)
os.makedirs(DST_DIR)
shutil.copytree(os.path.join(SRC_DIR, 'cg'), os.path.join(DST_DIR, 'cg'))
for fn in ('unknown0_lgbm.txt', 'unknown0_lgbm_prep.json', 'deck.csv'):
    sp = os.path.join(SRC_DIR, fn)
    if os.path.exists(sp): shutil.copy(sp, os.path.join(DST_DIR, fn))
with open(os.path.join(DST_DIR, 'main.py'), 'w') as f:
    f.write(src)
import py_compile
py_compile.compile(os.path.join(DST_DIR, 'main.py'), doraise=True)
print('compiles OK (W=%s)' % W)
shutil.rmtree(os.path.join(DST_DIR, '__pycache__'), ignore_errors=True)
with tarfile.open(TAR, 'w:gz') as tf:
    for fn in ('main.py', 'deck.csv', 'unknown0_lgbm.txt', 'unknown0_lgbm_prep.json'):
        tf.add(os.path.join(DST_DIR, fn), arcname=fn)
    cgd = os.path.join(DST_DIR, 'cg')
    for fn in os.listdir(cgd):
        if 'Zone.Identifier' in fn or fn == '__pycache__': continue
        tf.add(os.path.join(cgd, fn), arcname=os.path.join('cg', fn))
print('wrote', TAR)
