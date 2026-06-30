"""v10d2 = v09d4 forward-sim (ctx0, 1-ply) + IMITATION TIEBREAK gating.

Diagnostic: ~33% of ctx0 decisions have near-tied sim values (sim can't distinguish);
v09d4 picks ~arbitrarily there. On those, defer to the LGBM expert imitation.
Gate: pick the sim-best option; among options within EPS of the sim-best (a sim "tie"),
pick the one with the highest LGBM imitation score. Decisive sim lines are unchanged.

Low-correlation ensemble of two paradigms (calculation vs imitation), opponent-agnostic
(it's about MY decision-making) so mirror self-play can partially validate it.

Usage: build_v10d2_submission.py <srcdir> <dstdir> <tar> <EPS>
"""
import os, shutil, tarfile, sys

SRC_DIR = sys.argv[1] if len(sys.argv) > 1 else '/tmp/realagent'
DST_DIR = sys.argv[2] if len(sys.argv) > 2 else '/tmp/agent_v10d2'
TAR = sys.argv[3] if len(sys.argv) > 3 else '/kaggle/working/pokemon-20260627-v0-10d2-gate.tar.gz'
EPS = float(sys.argv[4]) if len(sys.argv) > 4 else 20.0

with open(os.path.join(SRC_DIR, 'main.py')) as f:
    src = f.read()
assert src.count('def agent(obs_dict') == 1
src = src.replace('def agent(obs_dict', 'def _base_agent(obs_dict')

ROLLOUT = ('\n_GATE_EPS = ' + repr(EPS) + '\n') + r'''
# ===================== v10d2 forward-sim + IMITATION-TIEBREAK gate =============
import ctypes as _ctypes
import json as _json
import cg.api as _rapi
from cg.sim import lib as _rlib


def _g_raw_step(search_id, select):
    arr = (_ctypes.c_int * len(select))(*select)
    bs = _rlib.SearchStep(_rapi.agent_ptr, search_id, arr, len(select))
    return _json.loads(bs.decode('utf-8') if isinstance(bs, (bytes, bytearray)) else bs)


def _g_hp(pl):
    t = 0
    for p in (pl.get('active') or []):
        if p: t += int(p.get('hp', 0) or 0)
    for p in (pl.get('bench') or []):
        if p: t += int(p.get('hp', 0) or 0)
    return t


def _g_eval(obs, p):
    cur = obs.get('current') if obs else None
    if cur is None: return -1e9
    me, op = cur['players'][p], cur['players'][1 - p]
    my_prize = len(me.get('prize') or []); op_prize = len(op.get('prize') or [])
    oa = (op.get('active') or [None]); op_hp = int(oa[0].get('hp', 0)) if (oa and oa[0]) else 0
    my_hand = me.get('handCount', 0) or len(me.get('hand') or [])
    return 1000.0 * (op_prize - my_prize) - 2.0 * op_hp + (_g_hp(me) - _g_hp(op)) + 5.0 * my_hand


def _g_rollout(child, p, cap=60):
    node = child
    for _ in range(cap):
        st = node.get('state', node); obs = st.get('observation'); sid = st.get('searchId')
        cur = obs.get('current') if obs else None
        if cur is None or cur.get('result', -1) != -1 or cur.get('yourIndex') != p:
            return _g_eval(obs, p)
        if obs.get('select') is None: return _g_eval(obs, p)
        try: ch = _base_agent(obs)
        except Exception: return _g_eval(obs, p)
        if ch is None: return _g_eval(obs, p)
        try:
            node = _g_raw_step(sid, ch)
            if node.get('error', 0) != 0: return _g_eval(obs, p)
        except Exception: return _g_eval(obs, p)
    return _g_eval(node.get('state', node).get('observation'), p)


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
        sims = []
        for oi in range(n):
            try:
                child = _g_raw_step(rid, [oi])
                sims.append(_g_rollout(child, p) if child.get('error', 0) == 0 else -1e17)
            except Exception:
                sims.append(-1e17)
        try:
            _rapi.search_end()
        except Exception:
            pass
        best = max(sims)
        near = [oi for oi in range(n) if best - sims[oi] <= _GATE_EPS]
        if len(near) <= 1:
            return [int(max(range(n), key=lambda i: sims[i]))]
        # sim is flat across `near` -> defer to LGBM expert imitation among them
        try:
            lg = _u0_lgbm_scores(ob, ob.select, p, 1 - p)
        except Exception:
            lg = None
        if lg is not None and len(lg) == n:
            return [int(max(near, key=lambda i: lg[i]))]
        return [int(max(range(n), key=lambda i: sims[i]))]
    except Exception:
        return _base_agent(obs_dict)
    finally:
        g['pre_turn'], g['ability_used_dudunsparce'], g['ability_used_fezandipiti'] = snap
# =================== end v10d2 forward-sim + IMITATION-TIEBREAK gate ===========
'''

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
print('main.py compiles OK (EPS=%s)' % EPS)

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
