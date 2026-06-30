"""v12d de-risk: is Archaludon forward-sim-friendly (playable WITHOUT a rule base)?

generic_fwdsim (deck-agnostic forward-sim, NO rule base) scored only 0.19 vs the
Alakazam champion when playing Alakazam (control deck = forward-sim-hostile).
Here we run the SAME generic_fwdsim but playing the ARCHALUDON deck, vs the Alakazam
agents. If it scores MUCH higher than 0.19, Archaludon is forward-sim-native and a
no-rule-base agent (forward-sim + imitation) is viable -> invest in the full build.
"""
import sys, time, json, importlib.util
sys.path.insert(0, '/kaggle/input/competitions/pokemon-tcg-ai-battle/sample_submission')
import cg.game as game
import generic_fwdsim_test as G   # has generic_fwdsim(obs,deck), V4 (v09d4), ALAK

ARCH = json.load(open('/kaggle/working/deck_archaludon.json'))
ALAK = G.ALAK

def load(d, name):
    sys.path.insert(0, d)
    sp = importlib.util.spec_from_file_location(name, d + '/main.py')
    m = importlib.util.module_from_spec(sp); sys.modules[name] = m; sp.loader.exec_module(m)
    return m
CH = load('/tmp/agent_v11d1_28thfull', 'champ')  # Alakazam champion

def fv(o):
    sel = o.get('select')
    if sel is None: return None
    mn = int(sel.get('minCount', 1) or 1); n = len(sel.get('option', []) or [])
    return list(range(min(max(1, mn), n))) or [0]

def gfs_arch(o):  # generic forward-sim playing Archaludon
    return G.generic_fwdsim(o, ARCH)

def champ_pol(o):
    try: return CH.agent(o)
    except Exception: return fv(o)

def v9_pol(o):
    try: return G.V4.agent(o)
    except Exception: return fv(o)

def play(A_pol, A_deck, B_pol, B_deck, A_is_p0):
    if hasattr(CH, 'pre_turn'): CH.pre_turn = -1
    if hasattr(G.V4, 'pre_turn'): G.V4.pre_turn = -1
    deck0, deck1 = (A_deck, B_deck) if A_is_p0 else (B_deck, A_deck)
    obs, sd = game.battle_start(deck0, deck1)
    if obs is None: return None
    for _ in range(2000):
        cur = obs.get('current')
        if cur is not None and cur.get('result', -1) != -1:
            return cur['result']
        if obs.get('select') is None: return None
        pp = cur['yourIndex'] if cur else 0
        a_turn = (pp == 0) == A_is_p0
        ch = A_pol(obs) if a_turn else B_pol(obs)
        if ch is None: return None
        try: obs = game.battle_select(ch)
        except Exception: return None
    return None

def match(name, A_pol, A_deck, B_pol, B_deck, N):
    aw = bw = dr = er = 0; t0 = time.perf_counter()
    for g in range(N):
        a_is_p0 = (g % 2 == 0)
        r = play(A_pol, A_deck, B_pol, B_deck, a_is_p0)
        game.battle_finish()
        if r is None: er += 1; continue
        a_idx = 0 if a_is_p0 else 1
        if r == 2: dr += 1
        elif r == a_idx: aw += 1
        else: bw += 1
    tot = aw + bw + dr; wr = aw / tot if tot else 0
    std = (wr * (1 - wr) / tot) ** 0.5 if tot else 0
    print(f'[{name}] A(Archaludon)={aw} B(Alakazam)={bw} draw={dr} err={er} | Arch_winrate={wr:.3f} (+-{1.96*std:.3f}) | {time.perf_counter()-t0:.1f}s')

if __name__ == '__main__':
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    print(f'=== v12d de-risk: generic_fwdsim(Archaludon) vs Alakazam agents, N={N} ===')
    print('(reference: generic_fwdsim(Alakazam) vs v09d4 = 0.19)')
    match('gfs(Archaludon) vs v09d4', gfs_arch, ARCH, v9_pol, ALAK, N)
    match('gfs(Archaludon) vs v11d1-28thfull(champ)', gfs_arch, ARCH, champ_pol, ALAK, N)
