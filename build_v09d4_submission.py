"""Build v09d4 submission = v08d34 agent + ROLLOUT forward-sim for UNKNOWN_0.

Confirmed (N=600): rollout vs v08d34 = 0.548 (+-0.040 95%CI) — significant win.

This injects the rollout wrapper into v08d34's submission main.py:
  - rename original `def agent(` -> `def _base_agent(`
  - append rollout helpers + a new `agent()` that, for UNKNOWN_0 (ctx==0) multi-
    option single-pick decisions, forward-simulates each first option and rolls
    out the rest of the turn using _base_agent as the rollout policy, evaluating
    the end-of-turn board; picks the best first option. Everything else -> _base_agent.
  - global snapshot/restore around the rollout so _base_agent's per-turn globals
    (pre_turn, ability_used_*) mutated during rollout don't corrupt the live game.

Packages: main.py + unknown0_lgbm.txt + unknown0_lgbm_prep.json + cg/ into
/tmp/agent_v09d4 and a tar at /kaggle/working/pokemon-20260627-v0-09d4-submission-rollout.tar.gz
"""
import os, shutil, tarfile

import sys as _sys
SRC_DIR = _sys.argv[1] if len(_sys.argv) > 1 else '/tmp/realagent'   # base agent dir (main.py+lgbm+cg)
DST_DIR = _sys.argv[2] if len(_sys.argv) > 2 else '/tmp/agent_v09d4'
TAR = _sys.argv[3] if len(_sys.argv) > 3 else '/kaggle/working/pokemon-20260627-v0-09d4-submission-rollout.tar.gz'
CTX_MODE = _sys.argv[4] if len(_sys.argv) > 4 else 'u0'              # 'u0' = ctx==0 only, 'all' = all decisions
DEPTH = int(_sys.argv[5]) if len(_sys.argv) > 5 else 1               # 1 = my turn only, 2 = through opponent's turn
EVAL_MODE = _sys.argv[6] if len(_sys.argv) > 6 else 'v1'             # 'v1' base eval, 'v2' lethal-setup aware
BENCH_INS = (_sys.argv[7].lower() in ('1', 'true', 'on')) if len(_sys.argv) > 7 else False  # v11d2 bench-insurance
EX_GUARD = (_sys.argv[8].lower() in ('1', 'true', 'on')) if len(_sys.argv) > 8 else False  # v11d4 ex-exposure guard
CLOCK = (_sys.argv[9].lower() in ('1', 'true', 'on')) if len(_sys.argv) > 9 else False  # v13d clock race-urgency
TAKEWIN = (_sys.argv[10].lower() in ('1', 'true', 'on')) if len(_sys.argv) > 10 else False  # v13d-2 rollout take-immediate-win
STEP2 = (_sys.argv[11].lower() in ('1', 'true', 'on')) if len(_sys.argv) > 11 else False  # v13d3 2-step ctx0 optimization (no pruning)
REDET = (_sys.argv[12].lower() in ('1', 'true', 'on')) if len(_sys.argv) > 12 else False  # v13d4 re-determinization race-aware aggression
MC_SAMPLES = int(_sys.argv[13]) if len(_sys.argv) > 13 else 1  # v16d Monte Carlo: K re-determinized rollout samples averaged
import json as _json
_THREAT_TBL = _json.load(open('/kaggle/working/archetype_threat.json')) if REDET else {}

with open(os.path.join(SRC_DIR, 'main.py')) as f:
    src = f.read()

assert 'def agent(obs_dict' in src, 'agent def not found'
assert src.count('def agent(obs_dict') == 1, 'multiple agent defs'
src = src.replace('def agent(obs_dict', 'def _base_agent(obs_dict')

if CTX_MODE == 'u0':
    _CTXS_LITERAL = '{0}'
elif CTX_MODE == 'all':
    _CTXS_LITERAL = 'None'   # None = all contexts
else:
    _CTXS_LITERAL = '{' + ', '.join(CTX_MODE.split(',')) + '}'   # e.g. "0,7" -> {0, 7}

ROLLOUT = ('\n_ROLLOUT_CTXS = ' + _CTXS_LITERAL
           + '\n_ROLLOUT_DEPTH = ' + str(DEPTH)
           + '\n_ROLLOUT_CAP = ' + ('120' if DEPTH >= 2 else '60')
           + "\n_ROLLOUT_EVAL = '" + EVAL_MODE + "'"
           + '\n_V11_BENCH = ' + ('True' if BENCH_INS else 'False')
           + '\n_V11_EX = ' + ('True' if EX_GUARD else 'False')
           + '\n_V13_CLOCK = ' + ('True' if CLOCK else 'False')
           + '\n_V13_TAKEWIN = ' + ('True' if TAKEWIN else 'False')
           + '\n_V13_STEP2 = ' + ('True' if STEP2 else 'False')
           + '\n_V13_REDET = ' + ('True' if REDET else 'False')
           + '\n_MC_SAMPLES = ' + str(MC_SAMPLES)
           + '\n_THREAT = ' + repr(_THREAT_TBL) + '\n') + r'''

# ========================= v09d4 ROLLOUT forward-sim ==========================
import ctypes as _ctypes
import json as _json
import cg.api as _rapi
from cg.sim import lib as _rlib

# v10d: attack id -> (damage, energy_cost) for the opponent-threat defensive eval
try:
    from cg.api import all_attack as _all_attack10
    _ATK10 = {a.attackId: (int(getattr(a, 'damage', 0) or 0), len(getattr(a, 'energies', None) or []))
              for a in _all_attack10()}
except Exception:
    _ATK10 = {}


def _rollout_raw_step(search_id, select):
    arr = (_ctypes.c_int * len(select))(*select)
    bs = _rlib.SearchStep(_rapi.agent_ptr, search_id, arr, len(select))
    return _json.loads(bs.decode('utf-8') if isinstance(bs, (bytes, bytearray)) else bs)


def _rollout_hp(pl):
    t = 0
    for p in (pl.get('active') or []):
        if p:
            t += int(p.get('hp', 0) or 0)
    for p in (pl.get('bench') or []):
        if p:
            t += int(p.get('hp', 0) or 0)
    return t


# v15d: learned value (GBM on REAL top-player game outcomes, AUC 0.80). Supplies the
# multi-turn positional foresight the 1-ply hand-eval is blind to (deckout/board/tempo).
_V15_VALUE = None
def _v15_load_value():
    global _V15_VALUE
    if _V15_VALUE is None:
        import lightgbm as _lgb, os as _os
        _V15_VALUE = _lgb.Booster(model_file=_os.path.join(_os.path.dirname(__file__), 'v15d_value.txt'))
    return _V15_VALUE


def _v15_pwin(me, op, cur):
    import numpy as _np
    def _poks(pl):
        a = pl.get('active') or []
        return [x for x in a if x] + [b for b in (pl.get('bench') or []) if b]
    ma = me.get('active') or [None]; m_act = ma[0] if ma else None
    oa = op.get('active') or [None]; o_act = oa[0] if oa else None
    mp = _poks(me)
    def _cnt(cid): return sum(1 for x in mp if x.get('id') == cid)
    mh = int(me.get('handCount', 0) or 0)
    feats = [int(cur.get('turn', 0) or 0),
             int(m_act.get('hp', 0) or 0) if m_act else -1,
             len(m_act.get('energies') or []) if m_act else -1,
             int(o_act.get('hp', 0) or 0) if o_act else -1,
             len(o_act.get('energies') or []) if o_act else -1,
             len([b for b in (me.get('bench') or []) if b]),
             len([b for b in (op.get('bench') or []) if b]),
             _cnt(743), _cnt(742), _cnt(741), _cnt(66),
             mh, int(op.get('handCount', 0) or 0),
             int(me.get('deckCount', 0) or 0), int(op.get('deckCount', 0) or 0),
             len(me.get('prize') or []), len(op.get('prize') or []), mh * 20]
    try:
        return float(_v15_load_value().predict(_np.array([feats], dtype=float))[0])
    except Exception:
        return 0.5


def _rollout_eval(obs, p):
    cur = obs.get('current') if obs else None
    if cur is None:
        return -1e9
    # v11d1: TERMINAL-STATE AWARENESS (universal correctness). The sim already
    # produces result (winner index); value a game-ending state explicitly so the
    # agent SEIZES wins (e.g. KO opponent's last Pokemon = board collapse, ~47% of
    # decided games) and AVOIDS in-turn self-losses. Not a tuned weight -> transfers.
    _res = cur.get('result', -1)
    if _res == p:
        return 1e7
    if _res == (1 - p):
        return -1e7
    if _res == 2:
        return -1e5
    pls = cur['players']
    me, op = pls[p], pls[1 - p]
    my_prize = len(me.get('prize') or [])
    op_prize = len(op.get('prize') or [])
    oa = (op.get('active') or [None])
    op_hp = int(oa[0].get('hp', 0)) if (oa and oa[0]) else 0
    my_hand = me.get('handCount', 0) or len(me.get('hand') or [])
    base = (1000.0 * (op_prize - my_prize) - 2.0 * op_hp
            + (_rollout_hp(me) - _rollout_hp(op)))
    if _V13_REDET:
        # v13d4: RE-DETERMINIZATION race-aware aggression (OFFENSE). Predict the opponent's
        # haymaker EARLY from their ARCHETYPE's deck-pool distribution (threat table), not
        # just the visible active. If that predicted threat will be online soon (clock<=1)
        # and can KO my active, RACE: punch for prizes NOW (drop setup/board-preservation)
        # to win the punch-out before they do. Mirror opp = Alakazam (low 90 threat, < my
        # HP) -> rarely fires -> non-breaking; the value is vs the diverse field
        # (Lucario 270@2, generic 220@3) -> Kaggle-judged.
        _opoks = ([oa[0]] if (oa and oa[0]) else []) + [b for b in (op.get('bench') or []) if b]
        _oids = {_pk.get('id') for _pk in _opoks}
        if _oids & {Hop_Phantump, Hop_Trevenant, Hop_Cramorant, Hop_Snorlax}:
            _arch = 'hop_control'
        elif _oids & {Riolu, Mega_Lucario_ex}:
            _arch = 'lucario'
        elif _oids & {Abra, Kadabra, Alakazam, Dunsparce, Dudunsparce}:
            _arch = 'alakazam_mirror'
        else:
            _arch = 'generic_control'
        _adng, _acost = _THREAT.get(_arch, _THREAT.get('_field', [200, 3]))
        _charge = max([len(_pk.get('energies') or []) for _pk in _opoks] or [0])
        _pred_clock = _acost - _charge
        _ma = me.get('active') or [None]
        _myhp = int(_ma[0].get('hp', 0) or 0) if (_ma and _ma[0]) else 0
        if _pred_clock <= 1 and _myhp > 0 and _adng >= _myhp:
            return 1000.0 * (op_prize - my_prize) - 3.0 * op_hp
    if _V13_CLOCK:
        # v13d: NOW-OR-NEVER race urgency (OFFENSE, not defense). If the opponent's
        # strong attack is online (clock<=1 = the measured 0.466-danger zone), there's
        # no time to set up -> value ONLY immediate prizes + damage to opponent (drop
        # the future-setup hand term and my-board preservation). Pushes AGGRESSION when
        # threatened (opposite of ex-guard's passivity). Observable (energies + real
        # attack data), not a tuned penalty.
        _oclk = 99
        _opoks = ([oa[0]] if (oa and oa[0]) else []) + [b for b in (op.get('bench') or []) if b]
        for _opk in _opoks:
            _e = len(_opk.get('energies') or [])
            _cd = card_table.get(_opk.get('id'))
            if _cd is None:
                continue
            for _aid in (getattr(_cd, 'attacks', None) or []):
                _dmg, _cost = _ATK10.get(_aid, (0, 99))
                if _dmg >= 60:  # a "strong" attack
                    _c = _cost - _e
                    if _c < _oclk:
                        _oclk = _c
        if _oclk <= 1:
            return 1000.0 * (op_prize - my_prize) - 3.0 * op_hp
    if _V11_BENCH:
        # v11d2: board-collapse precursor (top loss mode ~47%). Penalize ending the
        # turn with NO promotable bench; severe if my active is also KO-able next turn
        # by the opponent's VISIBLE active's known attack (real card data -> observable,
        # not an opponent-deck assumption).
        _bench_n = len([b for b in (me.get('bench') or []) if b])
        if _bench_n == 0:
            _ma = (me.get('active') or [None]); _myact = _ma[0] if _ma else None
            if _myact is None:
                base -= 800.0
            else:
                _myhp = int(_myact.get('hp', 0) or 0)
                _opa = oa[0] if oa else None
                _threat = 0
                if _opa is not None:
                    _ope = len(_opa.get('energies') or [])
                    _cd = card_table.get(_opa.get('id'))
                    if _cd is not None:
                        for _aid in (getattr(_cd, 'attacks', None) or []):
                            _dmg, _cost = _ATK10.get(_aid, (0, 99))
                            if _cost <= _ope and _dmg > _threat:
                                _threat = _dmg
                base -= 800.0 if (_myhp > 0 and _threat >= _myhp) else 150.0
        # deckout-avoidance (conservative: only imminent, ~8% loss mode). Keep mild so
        # it does NOT fight the Alakazam draw engine (drawing deep is core to Powerful Hand).
        _mydeck = int(me.get('deckCount', 0) or 0)
        if _mydeck <= 2:
            base -= 400.0 * (3 - _mydeck)
        # offensive collapse-press: a benchless opponent is one KO from board collapse
        # (the ~47% win mode). Mildly prefer pressuring them (terminal-awareness already
        # seizes the actual KO-win; this biases setup toward it).
        _opbench = len([b for b in (op.get('bench') or []) if b])
        if _opbench == 0:
            base += 120.0
    if _V11_EX:
        # v11d4: PRIZE-RACE defense (dominant loss mode ~85%). Penalize ending the turn
        # with my ACTIVE being a 2-prize EX Pokemon that is KO-able next turn by the
        # opponent's VISIBLE active's known attack -> giving up 2 prizes accelerates the
        # opponent's race. Observable (card_table.ex + real attack data), opponent-agnostic.
        _ma = (me.get('active') or [None]); _myact = _ma[0] if _ma else None
        if _myact is not None:
            _cdme = card_table.get(_myact.get('id'))
            if _cdme is not None and getattr(_cdme, 'ex', False):
                _myhp = int(_myact.get('hp', 0) or 0)
                _opa = oa[0] if oa else None
                _threat = 0
                if _opa is not None:
                    _ope = len(_opa.get('energies') or [])
                    _cdop = card_table.get(_opa.get('id'))
                    if _cdop is not None:
                        for _aid in (getattr(_cdop, 'attacks', None) or []):
                            _dmg, _cost = _ATK10.get(_aid, (0, 99))
                            if _cost <= _ope and _dmg > _threat:
                                _threat = _dmg
                if _myhp > 0 and _threat >= _myhp:
                    base -= 500.0   # exposing a 2-prize EX to KO
    if _ROLLOUT_EVAL == 'v15d':
        # v15d: SEARCH x LEARNED VALUE. Keep terminal awareness (handled above) and the
        # dominant CAUSAL prize term (proven prize-seeking strength); REPLACE the hand-crafted
        # positional heuristics (-2*op_hp + hp-diff + 5*hand) with a value learned from REAL
        # top-player outcomes -> multi-turn positional foresight the 1-ply eval can't see.
        # 600*pwin in [0,600] < one prize (1000): prizes still dominate, value ranks ties.
        return 1000.0 * (op_prize - my_prize) + 600.0 * _v15_pwin(me, op, cur)
    if _ROLLOUT_EVAL == 'v15dp':
        # v15dp: PURE learned value drives the leaf (terminal still seizes wins above). The
        # value includes prizes among its features, so it should seek prizes AND add positional.
        # Decisive test: if value given FULL control still ~= hand-eval, eval quality is NOT
        # the bottleneck (the value's AUC is largely non-actionable confound).
        return 2000.0 * _v15_pwin(me, op, cur)
    if _ROLLOUT_EVAL == 'v2':
        # Alakazam win condition: Powerful Hand deals 20 dmg per card in hand.
        # Reward states set up for a lethal KO next turn; lighter raw hand weight.
        lethal = 50.0 if (op_hp > 0 and 20 * my_hand >= op_hp) else 0.0
        return base + lethal + 3.0 * my_hand
    if _ROLLOUT_EVAL == 'v4':
        # v10d: OPPONENT-AGNOSTIC defensive read. Using the opponent's VISIBLE active
        # and REAL card/attack data (no deck/hand assumption -> robust by construction,
        # cannot fail like v09d6's wrong-opponent simulation), penalize end-of-turn
        # states where the opponent's current active can KO my active next turn.
        pen = 0.0
        try:
            ma = (me.get('active') or [None]); my_act = ma[0] if ma else None
            opa = oa[0] if oa else None
            if my_act and opa:
                my_act_hp = int(my_act.get('hp', 0) or 0)
                op_e = len(opa.get('energies') or [])
                cd = card_table.get(opa.get('id'))
                threat = 0
                if cd is not None:
                    for aid in (getattr(cd, 'attacks', None) or []):
                        dmg, cost = _ATK10.get(aid, (0, 99))
                        if cost <= op_e and dmg > threat:
                            threat = dmg
                if my_act_hp > 0 and threat >= my_act_hp:
                    pen = 350.0   # my active is KO-able next turn -> opponent likely takes a prize
        except Exception:
            pen = 0.0
        return base + 5.0 * my_hand - pen
    if _ROLLOUT_EVAL == 'v14d1':
        # v14d1: POWERFUL-HAND ENGINE DISCIPLINE (deckout self-defeat = 25% of real losses).
        # The champion's `+5*my_hand` is UNCAPPED -> the forward-sim hoards cards (observed
        # 19-25 card hands) and decks ITSELF out while AHEAD. Fix (opponent-agnostic, pure
        # correctness -- deckout is a loss by rule, hoarding past lethal is worthless):
        #   (1) cap the hand reward (cards beyond ~10 are not value, just deckout risk),
        #   (2) when AHEAD/even on prizes, penalize a thinning deck (closing out > drawing).
        # When BEHIND, no penalty -> keep digging hard (racing strength preserved).
        _deck = int(me.get('deckCount', 0) or 0)
        _hand_reward = 5.0 * min(my_hand, 10)
        _pen = 0.0
        if my_prize <= op_prize and _deck <= 8:
            _pen = (9 - _deck) ** 2 * 6.0
        return base + _hand_reward - _pen
    if _ROLLOUT_EVAL == 'v14d1b':
        # v14d1b: STRONGER deckout discipline when AHEAD (v14d1 only cut deckout 47%->28%,
        # hand still ~19). When ahead/even, hoarding cards cannot take prizes faster -> cap
        # hand reward hard (6) and start penalizing deck depletion earlier (deck<=12) and
        # steeper, to force closing out. When BEHIND, dig hard for lethal (race preserved),
        # only avoiding literal deckout. Opponent-agnostic, pure correctness.
        _deck = int(me.get('deckCount', 0) or 0)
        if my_prize <= op_prize:
            _hand_reward = 5.0 * min(my_hand, 6)
            _pen = max(0, 12 - _deck) ** 2 * 5.0
        else:
            _hand_reward = 5.0 * min(my_hand, 12)
            _pen = (9 - _deck) ** 2 * 6.0 if _deck <= 8 else 0.0
        return base + _hand_reward - _pen
    if _ROLLOUT_EVAL == 'v14d4':
        # v14d4 (angle 9): BACKUP-ATTACKER READINESS / energy preservation (opponent-agnostic,
        # offense-supporting -- NOT ex-guard defense). Alakazam attacks on 1 energy (dmg=hand*20);
        # piling energy on the active is wasted AND dies with it when KO'd (energy scarce: 7/deck).
        # Reward having a READY bench attacker (psy line with >=1 energy) so a KO does not cost a
        # tempo turn re-setting up (the out-raced loss mode). Keeps the agent attacking through KOs.
        _bench = me.get('bench') or []
        _backup = sum(1 for b in _bench if b and b.get('id') in (Abra, Kadabra, Alakazam)
                      and len(b.get('energies') or []) >= 1)
        return base + 5.0 * my_hand + 40.0 * min(_backup, 1)
    if _ROLLOUT_EVAL == 'v14d5':
        # v14d5: COMBINE the two confirmed opponent-agnostic levers --
        #   deckout discipline (v14d1: cap hand reward; penalize thinning deck when AHEAD), and
        #   backup-attacker readiness (v14d4: reward a ready bench attacker so KOs don't cost tempo).
        # Both target distinct real loss modes (deckout-while-ahead #1; out-raced fragility).
        _deck = int(me.get('deckCount', 0) or 0)
        _hand_reward = 5.0 * min(my_hand, 10)
        _pen = (9 - _deck) ** 2 * 6.0 if (my_prize <= op_prize and _deck <= 8) else 0.0
        _bench = me.get('bench') or []
        _backup = sum(1 for b in _bench if b and b.get('id') in (Abra, Kadabra, Alakazam)
                      and len(b.get('energies') or []) >= 1)
        return base + _hand_reward - _pen + 40.0 * min(_backup, 1)
    if _ROLLOUT_EVAL == 'v14d4b':
        # v14d4b: stronger backup pull (W=70) + reward up to 2 ready backups (more KO redundancy)
        _bench = me.get('bench') or []
        _backup = sum(1 for b in _bench if b and b.get('id') in (Abra, Kadabra, Alakazam)
                      and len(b.get('energies') or []) >= 1)
        return base + 5.0 * my_hand + 70.0 * min(_backup, 2)
    return base + 5.0 * my_hand


def _rollout_value(child, p, cap=_ROLLOUT_CAP):
    node = child
    left_my_turn = False
    for _ in range(cap):
        st = node.get('state', node)
        obs = st.get('observation')
        sid = st.get('searchId')
        cur = obs.get('current') if obs else None
        if cur is None or cur.get('result', -1) != -1:
            return _rollout_eval(obs, p)
        yi = cur.get('yourIndex')
        if _ROLLOUT_DEPTH <= 1:
            if yi != p:                       # my turn ended -> evaluate
                return _rollout_eval(obs, p)
        else:                                 # depth 2: roll through opponent's turn
            if yi != p:
                left_my_turn = True
            elif left_my_turn:                # opponent responded, back to my turn -> evaluate
                return _rollout_eval(obs, p)
        if obs.get('select') is None:
            return _rollout_eval(obs, p)
        ch = None
        if _V13_TAKEWIN and yi == p:
            # v13d-2: pure-correctness — if an immediate game-WIN option exists at this
            # rollout step, take it (so the rollout never misses a winnable line that
            # _base_agent would skip). Only on MY turn, single-pick options.
            _sel = obs['select']; _n = len(_sel.get('option') or []); _mn = int(_sel.get('minCount', 1) or 1)
            if _mn <= 1 and _n >= 1:
                for _oi in range(_n):
                    try:
                        _t = _rollout_raw_step(sid, [_oi])
                        if _t.get('error', 0) == 0:
                            _tc = _t.get('state', _t).get('observation', {}).get('current')
                            if _tc is not None and _tc.get('result', -1) == p:
                                ch = [_oi]; break
                    except Exception:
                        pass
        if ch is None:
            try:
                ch = _base_agent(obs)         # rollout policy for whoever is to move
            except Exception:
                return _rollout_eval(obs, p)
        if ch is None:
            return _rollout_eval(obs, p)
        try:
            node = _rollout_raw_step(sid, ch)
            if node.get('error', 0) != 0:
                return _rollout_eval(obs, p)
        except Exception:
            return _rollout_eval(obs, p)
    return _rollout_eval(node.get('state', node).get('observation'), p)


def _is_ctx0_node(obs):
    cur = obs.get('current') if obs else None
    if cur is None or cur.get('result', -1) != -1 or obs.get('select') is None:
        return False
    sel = obs['select']
    return sel.get('context') == 0 and len(sel.get('option') or []) >= 2 and int(sel.get('minCount', 1) or 1) <= 1


def _ff_to_ctx0(node, p, cap=40):
    """Advance with _base_agent through non-ctx0 decisions until the next ctx0 branch
    point, my-turn-end, or terminal. Returns (node, is_terminal_for_eval)."""
    for _ in range(cap):
        st = node.get('state', node)
        obs = st.get('observation'); sid = st.get('searchId')
        cur = obs.get('current') if obs else None
        if cur is None or cur.get('result', -1) != -1 or cur.get('yourIndex') != p or obs.get('select') is None:
            return node, True
        if _is_ctx0_node(obs):
            return node, False
        try:
            ch = _base_agent(obs)
        except Exception:
            return node, True
        if ch is None:
            return node, True
        try:
            nxt = _rollout_raw_step(sid, ch)
            if nxt.get('error', 0) != 0:
                return node, True
            node = nxt['state']
        except Exception:
            return node, True
    return node, True


def _value_2step(child, p):
    """v13d3: optimize the SECOND ctx0 action too (no pruning), _base_agent for the rest.
    Fast-forward to the next ctx0 branch; optimize over its options; return the best
    end-of-turn value. Makes the first-action valuation reflect 2-step-optimal play."""
    ffn, term = _ff_to_ctx0(child['state'] if 'state' in child else child, p)
    if term:
        return _rollout_eval(ffn.get('state', ffn).get('observation'), p)
    st = ffn.get('state', ffn)
    obs = st.get('observation'); sid = st.get('searchId')
    n = len(obs['select'].get('option') or [])
    best = -1e18
    for oj in range(n):
        try:
            c2 = _rollout_raw_step(sid, [oj])
            if c2.get('error', 0) != 0:
                continue
            v = _rollout_value(c2, p)
        except Exception:
            v = -1e17
        if v > best:
            best = v
    return best if best > -1e17 else _rollout_eval(obs, p)


def agent(obs_dict):
    sel = obs_dict.get('select')
    if not sel:
        return _base_agent(obs_dict)
    ctx = sel.get('context')
    n = len(sel.get('option') or [])
    mn = int(sel.get('minCount', 1) or 1)
    if (n < 2 or mn > 1) or (_ROLLOUT_CTXS is not None and ctx not in _ROLLOUT_CTXS):
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
        yd = list(my_deck)
        yp = list(my_deck)[:max(1, len(me.prize))]
        od = list(my_deck)
        op_ = list(my_deck)[:max(1, len(opp.prize))]
        oh = list(my_deck)[:max(1, opp.handCount)]
        oa = [741] if (len(opp.active) > 0 and opp.active[0] is None) else []
        # v16d: MONTE CARLO. The rollout is stochastic (re-determinized draws change the
        # eval ranking; a single sample makes an unstable, draw-luck-dependent choice). Run
        # K re-determinized search_begin passes and pick the option with the best AVERAGE
        # (expected) value -> correct decision under draw uncertainty. K=1 == prior behavior.
        _sums = [0.0] * n
        _cnts = [0] * n
        for _mk in range(_MC_SAMPLES):
            root = _rapi.search_begin(ob, yd, yp, od, op_, oh, oa)
            rid = root.searchId
            for oi in range(n):
                try:
                    child = _rollout_raw_step(rid, [oi])
                    if child.get('error', 0) != 0:
                        continue
                    v = _value_2step(child, p) if _V13_STEP2 else _rollout_value(child, p)
                except Exception:
                    v = -1e17
                _sums[oi] += v
                _cnts[oi] += 1
            try:
                _rapi.search_end()
            except Exception:
                pass
        best_i, best_v = 0, -1e18
        for oi in range(n):
            _m = (_sums[oi] / _cnts[oi]) if _cnts[oi] else -1e18
            if _m > best_v:
                best_v, best_i = _m, oi
        return [best_i]
    except Exception:
        return _base_agent(obs_dict)
    finally:
        g['pre_turn'], g['ability_used_dudunsparce'], g['ability_used_fezandipiti'] = snap
# ======================= end v09d4 ROLLOUT forward-sim =======================
'''

src = src + ROLLOUT

# write new agent dir
if os.path.exists(DST_DIR):
    shutil.rmtree(DST_DIR)
os.makedirs(DST_DIR)
shutil.copytree(os.path.join(SRC_DIR, 'cg'), os.path.join(DST_DIR, 'cg'))
for fn in ('unknown0_lgbm.txt', 'unknown0_lgbm_prep.json', 'deck.csv'):
    sp = os.path.join(SRC_DIR, fn)
    if os.path.exists(sp):
        shutil.copy(sp, os.path.join(DST_DIR, fn))
# v15d: learned-value model (trained on real-log outcomes) for the forward-sim leaf eval
_vp = '/kaggle/working/v15d_value.txt'
if os.path.exists(_vp):
    shutil.copy(_vp, os.path.join(DST_DIR, 'v15d_value.txt'))
with open(os.path.join(DST_DIR, 'main.py'), 'w') as f:
    f.write(src)

# syntax check
import py_compile
py_compile.compile(os.path.join(DST_DIR, 'main.py'), doraise=True)
print('main.py compiles OK')

# package tar (flat structure like other submissions)
with tarfile.open(TAR, 'w:gz') as tf:
    for fn in os.listdir(DST_DIR):
        if fn == 'cg':
            continue
        tf.add(os.path.join(DST_DIR, fn), arcname=fn)
    cgd = os.path.join(DST_DIR, 'cg')
    for fn in os.listdir(cgd):
        tf.add(os.path.join(cgd, fn), arcname=os.path.join('cg', fn))
print('wrote', TAR)
print('v09d4 build done')
