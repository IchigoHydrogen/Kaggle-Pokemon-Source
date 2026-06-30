"""Track B feasibility spike — can we forward-simulate inside inference?

Goal: prove that the competition engine's search API (search_begin/search_step)
can be driven from an agent observation, and measure latency per ply.
If latency per option is small enough (< a few ms), a 1-ply lookahead over
~5-10 options per UNKNOWN_0 decision is viable within Kaggle time limits.

This runs ENTIRELY locally (in the container) — no Kaggle submission needed.

Plan:
  1. import cg (engine) from the competition sample_submission dir
  2. battle_start(my_deck, my_deck) — a real Alakazam mirror
  3. drive with first-valid selections until we reach a decision with >=2 options
  4. from that observation, call search_begin (predict opponent hidden info
     with deck-card fillers), then search_step on each option; time each
  5. verify the resulting observation differs (forward model actually advances)
  6. report latency stats and any errors
"""
import sys, os, time, json

CG_PARENT = '/kaggle/input/competitions/pokemon-tcg-ai-battle/sample_submission'
sys.path.insert(0, CG_PARENT)

my_deck = [741, 741, 741, 741, 742, 742, 742, 742, 743, 743, 743, 305, 305, 305, 66, 66, 66, 140, 1231, 1231, 1231, 1231, 1225, 1225, 1225, 1225, 1182, 1182, 1182, 1184, 1184, 1086, 1086, 1086, 1086, 1152, 1152, 1152, 1152, 1079, 1079, 1079, 1081, 1081, 1081, 1081, 1129, 1097, 1156, 1174, 1266, 1266, 1266, 19, 19, 19, 19, 13, 5, 5]
assert len(my_deck) == 60

import cg.game as game
import cg.api as api
from cg.sim import Battle

print('=== engine import OK ===')

def first_valid_select(obs_dict):
    """Return a minimal valid select list for the current select."""
    sel = obs_dict.get('select')
    if sel is None:
        return None
    opts = sel.get('option', []) or []
    mn = sel.get('minCount', 1) or 1
    n = max(1, int(mn))
    idxs = list(range(min(n, len(opts))))
    return idxs if idxs else [0]

def n_options(obs_dict):
    sel = obs_dict.get('select')
    if sel is None:
        return 0
    return len(sel.get('option', []) or [])

def context_of(obs_dict):
    sel = obs_dict.get('select')
    if sel is None:
        return None
    return sel.get('context')

# 1. start battle
obs, sd = game.battle_start(my_deck, my_deck)
if obs is None:
    print('battle_start FAILED', sd.errorPlayer, sd.errorType)
    sys.exit(1)
print('=== battle_start OK ===')

# 2. drive until a decision with >=2 options, then test search
MAX_STEPS = 400
tested = 0
search_latencies = []
begin_latencies = []
for step in range(MAX_STEPS):
    nopt = n_options(obs)
    ctx = context_of(obs)
    if nopt >= 2 and tested < 5:
        # try a forward-search from THIS observation
        try:
            ob = api.to_observation_class(obs)
            st = ob.current
            yi = st.yourIndex
            opp = st.players[1 - yi]
            me = st.players[yi]
            # predicted hidden info — deck-card fillers (valid IDs, counts >= required)
            your_deck = list(my_deck)               # ignored if select.deck != None
            your_prize = list(my_deck)[:max(1, len(me.prize))]
            opp_deck = list(my_deck)                 # >= opp.deckCount
            opp_prize = list(my_deck)[:max(1, len(opp.prize))]
            opp_hand = list(my_deck)[:max(1, opp.handCount)]
            # face-down opponent active?
            opp_active = []
            act = opp.active
            if len(act) > 0 and act[0] is None:
                opp_active = [741]  # a basic pokemon id
            t0 = time.perf_counter()
            root = api.search_begin(ob, your_deck, your_prize,
                                    opp_deck, opp_prize, opp_hand, opp_active)
            t1 = time.perf_counter()
            begin_latencies.append((t1 - t0) * 1000)
            rid = root.searchId
            # step each option (1-ply), measure latency
            sel = obs['select']
            mn = sel.get('minCount', 1) or 1
            for oi in range(nopt):
                try:
                    s0 = time.perf_counter()
                    nxt = api.search_step(rid, [oi] if mn <= 1 else list(range(mn)))
                    s1 = time.perf_counter()
                    search_latencies.append((s1 - s0) * 1000)
                except Exception as e:
                    print(f'  search_step opt={oi} err: {e!r}')
                    break
            api.search_end()
            tested += 1
            print(f'[step {step}] ctx={ctx} nopt={nopt} '
                  f'search_begin={begin_latencies[-1]:.2f}ms tested_options={nopt}')
        except Exception as e:
            print(f'[step {step}] ctx={ctx} nopt={nopt} search_begin FAILED: {e!r}')
            tested += 1  # don't retry forever
    # advance the live game
    sl = first_valid_select(obs)
    if sl is None:
        print(f'[step {step}] no select (terminal?) — stopping')
        break
    try:
        obs = game.battle_select(sl)
    except Exception as e:
        print(f'[step {step}] battle_select({sl}) err: {e!r} — stopping')
        break
    if tested >= 5:
        break

game.battle_finish()

# 3. report
print('\n=== TRACK B FEASIBILITY REPORT ===')
print(f'search_begin calls: {len(begin_latencies)}')
if begin_latencies:
    print(f'  search_begin latency ms: min={min(begin_latencies):.2f} '
          f'max={max(begin_latencies):.2f} mean={sum(begin_latencies)/len(begin_latencies):.2f}')
print(f'search_step calls: {len(search_latencies)}')
if search_latencies:
    sl = sorted(search_latencies)
    print(f'  search_step latency ms: min={sl[0]:.3f} max={sl[-1]:.3f} '
          f'mean={sum(sl)/len(sl):.3f} median={sl[len(sl)//2]:.3f}')
    # viability estimate: ~8 options/decision, 1-ply
    per_decision = (sum(begin_latencies)/len(begin_latencies) if begin_latencies else 0) + \
                   8 * (sum(search_latencies)/len(search_latencies))
    print(f'  est. 1-ply lookahead per UNKNOWN_0 decision (begin + 8 options): {per_decision:.2f}ms')
    print(f'  VERDICT: {"VIABLE" if per_decision < 200 else "RISKY (>200ms/decision)"}')
else:
    print('  NO successful search_step — forward sim not usable as-is')
print('=== END ===')
