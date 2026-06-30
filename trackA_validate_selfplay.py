"""Validate Track A on the TRUE metric: v09d2 agent vs v08d34 agent in self-play.

Both are rule+LGBM agents; they differ ONLY in the LGBM model:
  - v08d34: LGBM trained on REAL op_last_context, but inference feeds the op_energy
            proxy (train/serve skew; infer_top1=0.5098)
  - v09d2:  LGBM trained on the op_energy PROXY, matching inference
            (skew killed; infer_top1=0.5246, +0.0148)

If v09d2 beats v08d34 head-to-head (>0.50), the +0.0148 infer_top1 translates to
real win-rate gain → Track A is a genuine Kaggle improvement (no submission spent).
"""
import sys, time, importlib.util

CG_PARENT = '/kaggle/input/competitions/pokemon-tcg-ai-battle/sample_submission'
sys.path.insert(0, CG_PARENT)
import cg.game as game
import cg.api as api


def load_agent(dir_path, modname):
    sys.path.insert(0, dir_path)  # so its own cg/ + lgbm files resolve via __file__
    spec = importlib.util.spec_from_file_location(modname, dir_path + '/main.py')
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod
    spec.loader.exec_module(mod)
    return mod

A34 = load_agent('/tmp/realagent', 'agent_d34')      # v08d34
A92 = load_agent('/tmp/agent_v09d2', 'agent_d2')     # v09d2
print('d34 LGBM:', getattr(A34, '_U0_LGBM', None) is not None,
      '| d2 LGBM:', getattr(A92, '_U0_LGBM', None) is not None)

my_deck = A34.my_deck


def first_valid(obs_dict):
    sel = obs_dict.get('select')
    if sel is None:
        return None
    mn = int(sel.get('minCount', 1) or 1)
    n = len(sel.get('option', []) or [])
    idxs = list(range(min(max(1, mn), n)))
    return idxs if idxs else [0]


def make_policy(mod):
    def pol(obs_dict):
        try:
            return mod.agent(obs_dict)
        except Exception:
            return first_valid(obs_dict)
    return pol


def play_game(p0, p1, mods, max_steps=2000):
    for m in mods:
        if hasattr(m, 'pre_turn'):
            m.pre_turn = -1
    obs, sd = game.battle_start(my_deck, my_deck)
    if obs is None:
        return None, 'start_fail'
    pols = [p0, p1]
    for _ in range(max_steps):
        cur = obs.get('current')
        if cur is not None and cur.get('result', -1) != -1:
            return cur['result'], 'ok'
        sel = obs.get('select')
        if sel is None:
            return None, 'no_select'
        p = cur['yourIndex'] if cur else 0
        choice = pols[p](obs)
        if choice is None:
            return None, 'no_choice'
        try:
            obs = game.battle_select(choice)
        except Exception as e:
            return None, f'err:{e!r}'
    return None, 'max_steps'


def run_match(name, modA, modB, n_games):
    polA, polB = make_policy(modA), make_policy(modB)
    aw = bw = dr = er = 0
    t0 = time.perf_counter()
    for g in range(n_games):
        if g % 2 == 0:
            res, st = play_game(polA, polB, [modA, modB]); a_is = 0
        else:
            res, st = play_game(polB, polA, [modA, modB]); a_is = 1
        game.battle_finish()
        if st != 'ok' or res is None:
            er += 1; continue
        if res == 2: dr += 1
        elif res == a_is: aw += 1
        else: bw += 1
    dt = time.perf_counter() - t0
    tot = aw + bw + dr
    wr = aw / tot if tot else 0.0
    # binomial std for reference
    std = (wr * (1 - wr) / tot) ** 0.5 if tot else 0
    print(f'[{name}] A={aw} B={bw} draw={dr} err={er} | A_winrate={wr:.3f} '
          f'(+-{1.96*std:.3f} 95%CI) | {dt:.1f}s')
    return wr


if __name__ == '__main__':
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    print(f'=== Track A validation: v09d2 vs v08d34, N={N} ===')
    run_match('v09d2 vs v08d34 (>0.50 = Track A wins)', A92, A34, N)
