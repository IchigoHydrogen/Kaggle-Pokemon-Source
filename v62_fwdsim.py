"""v62 + forward-sim: champion's winning recipe (rule base + forward-sim + terminal)
applied to the v62 Lucario rule base. ctx0 = search each option, rollout with v62's
rule base, terminal-aware leaf eval; non-ctx0 = v62 directly. v62 globals saved/restored
around the search so rollout simulation doesn't pollute the real decision."""
import sys, importlib.util
sys.path.insert(0,'/kaggle/working')
import generic_fwdsim_test as G
sys.path.insert(0,'/kaggle/working/agent_v62lucario')
_sp=importlib.util.spec_from_file_location('v62','/kaggle/working/agent_v62lucario/main.py')
V=importlib.util.module_from_spec(_sp); _sp.loader.exec_module(V)
api=G.api; raw_step=G.raw_step; fv=G.fv
my_deck=list(V.my_deck)

def _save(): return (V.plan, V.pre_turn, V.ability_used)
def _restore(s): V.plan, V.pre_turn, V.ability_used = s

def ev_leaf(obs,p):
    cur=obs.get('current') if obs else None
    if cur is not None:
        r=cur.get('result',-1)
        if r==p: return 1e7
        if r==(1-p): return -1e7
        if r==2: return -1e5
    return G.ev(obs,p)

def v62_rollout(state,p,cap=80):
    node=state
    for _ in range(cap):
        st=node.get('state',node); obs=st.get('observation'); sid=st.get('searchId')
        cur=obs.get('current') if obs else None
        if cur is None or cur.get('result',-1)!=-1 or cur.get('yourIndex')!=p: return obs
        if obs.get('select') is None: return obs
        try: ch=V.agent(obs) or fv(obs)
        except Exception: ch=fv(obs)
        if not ch: ch=fv(obs)
        try:
            node=raw_step(sid,ch)
            if node.get('error',0)!=0: return obs
        except Exception: return obs
    return node.get('state',node).get('observation')

def agent(obs_dict):
    sel=obs_dict.get('select')
    if sel is None: return V.agent(obs_dict)         # deck selection etc.
    n=len(sel.get('option') or []); mn=int(sel.get('minCount',1) or 1)
    ctx=sel.get('context')
    if ctx!=0 or n<2 or mn>1:
        return V.agent(obs_dict)                     # non-ctx0 / trivial -> v62 rule base
    saved=_save()
    try:
        ob=api.to_observation_class(obs_dict); stt=ob.current
        if stt is None or getattr(ob,'search_begin_input',None) is None: return V.agent(obs_dict)
        p=stt.yourIndex; me=stt.players[p]; opp=stt.players[1-p]
        yd=list(my_deck); yp=list(my_deck)[:max(1,len(me.prize))]; od=list(my_deck)
        op_=list(my_deck)[:max(1,len(opp.prize))]; oh=list(my_deck)[:max(1,opp.handCount)]
        oa=[my_deck[0]] if (len(opp.active)>0 and opp.active[0] is None) else []
        root=api.search_begin(ob,yd,yp,od,op_,oh,oa); rid=root.searchId
        bi,bv=0,-1e18
        for oi in range(n):
            try:
                child=raw_step(rid,[oi])
                end=v62_rollout(child['state'],p) if child.get('error',0)==0 else None
                v=ev_leaf(end,p) if end else -1e17
            except Exception: v=-1e17
            if v>bv: bv,bi=v,oi
        try: api.search_end()
        except Exception: pass
        return [bi]
    except Exception:
        return V.agent(obs_dict)
    finally:
        _restore(saved)
