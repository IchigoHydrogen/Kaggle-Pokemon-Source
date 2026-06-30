"""Validate the v09d4 rollout-injected submission vs v08d34, head-to-head.

Confirms the injected agent() (rollout + global snapshot/restore) reproduces the
~0.548 win rate measured by the standalone harness, i.e. the integration is correct.
"""
import sys, time, importlib.util
CG_PARENT = '/kaggle/input/competitions/pokemon-tcg-ai-battle/sample_submission'
sys.path.insert(0, CG_PARENT)
import cg.game as game


def load_agent(dir_path, modname):
    sys.path.insert(0, dir_path)
    spec = importlib.util.spec_from_file_location(modname, dir_path + '/main.py')
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod
    spec.loader.exec_module(mod)
    return mod

V4 = load_agent('/tmp/agent_v09d4', 'agent_v09d4')   # rollout-injected
D34 = load_agent('/tmp/realagent', 'agent_d34')       # base
print('v09d4 LGBM:', getattr(V4, '_U0_LGBM', None) is not None,
      '| d34 LGBM:', getattr(D34, '_U0_LGBM', None) is not None)
my_deck = D34.my_deck


def fv(o):
    sel = o.get('select')
    if sel is None:
        return None
    mn = int(sel.get('minCount', 1) or 1)
    n = len(sel.get('option', []) or [])
    return list(range(min(max(1, mn), n))) or [0]


def mk(mod):
    def pol(o):
        try:
            return mod.agent(o)
        except Exception:
            return fv(o)
    return pol


def play(p0, p1, mods, max_steps=2000):
    for m in mods:
        if hasattr(m, 'pre_turn'):
            m.pre_turn = -1
    obs, sd = game.battle_start(my_deck, my_deck)
    if obs is None:
        return None
    pols = [p0, p1]
    for _ in range(max_steps):
        cur = obs.get('current')
        if cur is not None and cur.get('result', -1) != -1:
            return cur['result']
        if obs.get('select') is None:
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


def match(name, mA, mB, N):
    pA, pB = mk(mA), mk(mB)
    aw = bw = dr = er = 0
    t0 = time.perf_counter()
    for g in range(N):
        if g % 2 == 0:
            r = play(pA, pB, [mA, mB]); ais = 0
        else:
            r = play(pB, pA, [mA, mB]); ais = 1
        game.battle_finish()
        if r is None: er += 1; continue
        if r == 2: dr += 1
        elif r == ais: aw += 1
        else: bw += 1
    tot = aw + bw + dr
    wr = aw / tot if tot else 0
    std = (wr * (1 - wr) / tot) ** 0.5 if tot else 0
    print(f'[{name}] A={aw} B={bw} draw={dr} err={er} | wr={wr:.3f} (+-{1.96*std:.3f}) | {time.perf_counter()-t0:.1f}s')
    return wr


if __name__ == '__main__':
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    print(f'=== validate v09d4 (rollout-injected) vs v08d34, N={N} ===')
    match('v09d4 vs v08d34 (>0.50 = rollout integrated OK)', V4, D34, N)
