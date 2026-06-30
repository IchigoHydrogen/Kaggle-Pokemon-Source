"""Track B step 5 — ROLLOUT-based forward-sim (the 'done right' version).

Naive 1-ply greedy lost to the strong agent (0.133). The fix: don't evaluate a
single action with a crude heuristic — instead, for each candidate FIRST action,
ROLL OUT the rest of my turn using the STRONG agent itself as the rollout policy,
then evaluate the END-OF-TURN board. Pick the first action whose rollout ends best.

This couples search (compare first actions) with the agent's strategic knowledge
(rollout), and evaluates at end-of-turn (when the Alakazam Powerful-Hand payoff is
realized) instead of mid-turn.

Key trick: call lib.SearchStep directly to get the raw engine-format observation
dict (bypassing the dataclass), so we can feed it straight to the real agent.

Only applied at UNKNOWN_0 (ctx=0) decisions; everything else uses the real agent.
Measure: rollout_agent vs real. >0.50 = forward-sim finally adds value.
"""
import sys, time, ctypes, json
sys.path.insert(0, '/kaggle/input/competitions/pokemon-tcg-ai-battle/sample_submission')
sys.path.insert(0, '/tmp/realagent')  # FIRST: real agent (has main.py + cg/ + lgbm)
import cg.game as game
import cg.api as api
from cg.sim import lib
import main as real

print('real LGBM:', getattr(real, '_U0_LGBM', None) is not None)
my_deck = real.my_deck


def first_valid(o):
    sel = o.get('select')
    if sel is None:
        return None
    mn = int(sel.get('minCount', 1) or 1)
    n = len(sel.get('option', []) or [])
    return list(range(min(max(1, mn), n))) or [0]


def real_pol(o):
    try:
        return real.agent(o)
    except Exception:
        return first_valid(o)


def _raw_step(search_id, select):
    arr = (ctypes.c_int * len(select))(*select)
    bs = lib.SearchStep(api.agent_ptr, search_id, arr, len(select))
    return json.loads(bs.decode('utf-8') if isinstance(bs, (bytes, bytearray)) else bs)


def _hp(pl):
    t = 0
    for p in (pl.get('active') or []):
        if p: t += int(p.get('hp', 0) or 0)
    for p in (pl.get('bench') or []):
        if p: t += int(p.get('hp', 0) or 0)
    return t


def _eval(obs_dict, p):
    cur = obs_dict.get('current')
    if cur is None:
        return -1e9
    pls = cur['players']
    me, op = pls[p], pls[1 - p]
    my_prize = len([c for c in (me.get('prize') or [])])
    op_prize = len([c for c in (op.get('prize') or [])])
    # deck-aware: prizes dominate; reward op active low HP; value next-turn hand (Powerful Hand)
    op_active = (op.get('active') or [None])
    op_hp = int(op_active[0].get('hp', 0)) if op_active and op_active[0] else 0
    my_hand = me.get('handCount', 0) or (len(me.get('hand') or []))
    return (1000.0 * (op_prize - my_prize)
            - 2.0 * op_hp
            + 1.0 * (_hp(me) - _hp(op))
            + 5.0 * my_hand)


def _rollout_value(child, p, start_turn, cap=60):
    """Roll out the rest of p's turn with the real agent; return end-of-turn eval."""
    node = child
    for _ in range(cap):
        st = node.get('state', node)
        obs = st.get('observation')
        sid = st.get('searchId')
        cur = obs.get('current') if obs else None
        if cur is None:
            return _eval(obs, p)
        if cur.get('result', -1) != -1:
            return _eval(obs, p)
        if cur.get('yourIndex') != p:          # opponent to move → my turn ended
            return _eval(obs, p)
        sel = obs.get('select')
        if sel is None:
            return _eval(obs, p)
        choice = real_pol(obs)
        try:
            node = _raw_step(sid, choice)
            if node.get('error', 0) != 0:
                return _eval(obs, p)
        except Exception:
            return _eval(obs, p)
    return _eval(node.get('state', node).get('observation'), p)


def rollout_agent(obs_dict):
    sel = obs_dict.get('select')
    if sel is None:
        return None
    ctx = sel.get('context')
    n = len(sel.get('option', []) or [])
    mn = int(sel.get('minCount', 1) or 1)
    # only intervene on UNKNOWN_0 (ctx==0) multi-option single-pick decisions
    if ctx != 0 or n < 2 or mn > 1:
        return real_pol(obs_dict)
    try:
        ob = api.to_observation_class(obs_dict)
        stt = ob.current
        if stt is None or getattr(ob, 'search_begin_input', None) is None:
            return real_pol(obs_dict)
        p = stt.yourIndex
        me, opp = stt.players[p], stt.players[1 - p]
        yd = list(my_deck); yp = list(my_deck)[:max(1, len(me.prize))]
        od = list(my_deck); op_ = list(my_deck)[:max(1, len(opp.prize))]
        oh = list(my_deck)[:max(1, opp.handCount)]
        oa = [741] if (len(opp.active) > 0 and opp.active[0] is None) else []
        root = api.search_begin(ob, yd, yp, od, op_, oh, oa)
        rid = root.searchId
        start_turn = stt.turn
        best_i, best_v = 0, -1e18
        for oi in range(n):
            try:
                child = _raw_step(rid, [oi])
                if child.get('error', 0) != 0:
                    continue
                v = _rollout_value(child, p, start_turn)
            except Exception:
                v = -1e17
            if v > best_v:
                best_v, best_i = v, oi
        api.search_end()
        return [best_i]
    except Exception:
        return real_pol(obs_dict)


def play(p0, p1, max_steps=2000):
    if hasattr(real, 'pre_turn'):
        real.pre_turn = -1
    obs, sd = game.battle_start(my_deck, my_deck)
    if obs is None:
        return None
    pols = [p0, p1]
    for _ in range(max_steps):
        cur = obs.get('current')
        if cur is not None and cur.get('result', -1) != -1:
            return cur['result']
        sel = obs.get('select')
        if sel is None:
            return None
        pp = cur['yourIndex'] if cur else 0
        ch = pols[pp](obs)
        if ch is None:
            return None
        try:
            obs = game.battle_select(ch)
        except Exception:
            return None
    return None


def match(name, A, B, N):
    aw = bw = dr = er = 0
    t0 = time.perf_counter()
    for g in range(N):
        if g % 2 == 0:
            r = play(A, B); ais = 0
        else:
            r = play(B, A); ais = 1
        game.battle_finish()
        if r is None: er += 1; continue
        if r == 2: dr += 1
        elif r == ais: aw += 1
        else: bw += 1
    tot = aw + bw + dr
    wr = aw / tot if tot else 0
    std = (wr * (1 - wr) / tot) ** 0.5 if tot else 0
    print(f'[{name}] A={aw} B={bw} draw={dr} err={er} | wr={wr:.3f} '
          f'(+-{1.96*std:.3f}) | {time.perf_counter()-t0:.1f}s')
    return wr


if __name__ == '__main__':
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    print(f'=== Track B rollout, N={N} ===')
    match('rollout vs real (>0.50 = wins)', rollout_agent, real_pol, N)
