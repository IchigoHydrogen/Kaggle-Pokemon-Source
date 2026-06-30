"""Generate self-play data for a LEARNED VALUE FUNCTION (rollout eval replacement).

Play self-play games with the strong v09d4 agent on both sides. Whenever a turn
ends (yourIndex flips), snapshot the END-OF-TURN board features from the perspective
of the player who just moved, tagged with that player. At game end, label every
snapshot with whether its player won. Output a parquet of (features, won).

A model trained on this predicts P(win | end-of-my-turn state) — an opponent-agnostic
positional value to replace the hand-crafted rollout eval in v09d4.
"""
import sys, json, importlib.util
sys.path.insert(0, '/kaggle/input/competitions/pokemon-tcg-ai-battle/sample_submission')
sys.path.insert(0, '/tmp/agent_v09d4')   # FIRST: v09d4 (rollout) agent
import cg.game as game
import importlib.util as ilu

spec = ilu.spec_from_file_location('agent_v09d4', '/tmp/agent_v09d4/main.py')
A = ilu.module_from_spec(spec); sys.modules['agent_v09d4'] = A; spec.loader.exec_module(A)
my_deck = A.my_deck
print('v09d4 agent loaded, LGBM:', getattr(A, '_U0_LGBM', None) is not None)


def fv(o):
    sel = o.get('select')
    if sel is None: return None
    mn = int(sel.get('minCount', 1) or 1); n = len(sel.get('option', []) or [])
    return list(range(min(max(1, mn), n))) or [0]


def pol(o):
    try: return A.agent(o)
    except Exception: return fv(o)


def _hp(pl):
    t = 0
    for p in (pl.get('active') or []):
        if p: t += int(p.get('hp', 0) or 0)
    for p in (pl.get('bench') or []):
        if p: t += int(p.get('hp', 0) or 0)
    return t


def _ids(pl):
    out = []
    for p in (pl.get('active') or []):
        if p: out.append(p.get('id'))
    for p in (pl.get('bench') or []):
        if p: out.append(p.get('id'))
    return out


def features(cur, p):
    me, op = cur['players'][p], cur['players'][1 - p]
    ids = _ids(me)
    oa = (op.get('active') or [None]); op_hp = int(oa[0].get('hp', 0)) if (oa and oa[0]) else 0
    ma = (me.get('active') or [None]); my_hp = int(ma[0].get('hp', 0)) if (ma and ma[0]) else 0
    my_act_e = len(ma[0].get('energyCards') or []) if (ma and ma[0]) else 0
    return {
        'my_prize': len(me.get('prize') or []),
        'op_prize': len(op.get('prize') or []),
        'prize_diff': len(op.get('prize') or []) - len(me.get('prize') or []),
        'my_active_hp': my_hp,
        'op_active_hp': op_hp,
        'my_total_hp': _hp(me),
        'op_total_hp': _hp(op),
        'my_hand': me.get('handCount', 0) or len(me.get('hand') or []),
        'op_hand': op.get('handCount', 0) or len(op.get('hand') or []),
        'my_deck': me.get('deckCount', 0),
        'op_deck': op.get('deckCount', 0),
        'my_bench': len(me.get('bench') or []),
        'op_bench': len(op.get('bench') or []),
        'my_active_energy': my_act_e,
        'my_alakazam': sum(1 for i in ids if i == 743),
        'my_kadabra': sum(1 for i in ids if i == 742),
        'my_abra': sum(1 for i in ids if i == 741),
        'turn': cur.get('turn', 0),
    }


def play_collect(snaps):
    A.pre_turn = -1
    obs, sd = game.battle_start(my_deck, my_deck)
    if obs is None:
        return None
    local = []   # (player, features)
    prev_player = None
    last_cur_by_player = {}
    for _ in range(2000):
        cur = obs.get('current')
        if cur is not None and cur.get('result', -1) != -1:
            winner = cur['result']
            for pl, feat in local:
                feat['won'] = 1 if (winner == pl) else 0
                snaps.append(feat)
            return winner
        if obs.get('select') is None:
            return None
        pp = cur['yourIndex'] if cur else 0
        # record a snapshot for the player about to act's CURRENT board each time the
        # acting player changes (captures end-of-prev-turn / start states across the game)
        if prev_player is not None and pp != prev_player and cur is not None:
            feat = features(cur, prev_player)
            local.append((prev_player, feat))
        prev_player = pp
        ch = pol(obs)
        if ch is None:
            return None
        try:
            obs = game.battle_select(ch)
        except Exception:
            return None
    return None


if __name__ == '__main__':
    import pandas as pd
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 150
    snaps = []
    nwin = 0
    for g in range(N):
        w = play_collect(snaps)
        game.battle_finish()
        if w in (0, 1):
            nwin += 1
    df = pd.DataFrame(snaps)
    out = '/kaggle/working/value_train_data.parquet'
    df.to_parquet(out)
    print(f'games={N} ok={nwin} snapshots={len(df)} -> {out}')
    if len(df):
        print('won rate in snaps:', round(df["won"].mean(), 3))
        print(df.describe().loc[['mean']].T.head(20))
