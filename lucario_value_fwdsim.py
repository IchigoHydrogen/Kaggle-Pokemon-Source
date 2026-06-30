"""Pilot = forward-sim + value-guided rollout (learned Lucario value) + terminal."""
import sys, lightgbm as lgb, numpy as np
sys.path.insert(0,'/kaggle/working')
import generic_fwdsim_test as G
api=G.api; raw_step=G.raw_step; fv=G.fv
VAL=lgb.Booster(model_file='/kaggle/working/lucario_value.txt')
MEGA=678; RIOLU=677; FIGHT=6
def feats(cur,p):
    me=cur['players'][p]; op=cur['players'][1-p]
    poks=([me['active'][0]] if me.get('active') and me['active'][0] else [])+[b for b in (me.get('bench') or []) if b]
    mega=[pk for pk in poks if pk.get('id')==MEGA]
    mfe=max([sum(1 for e in (pk.get('energies') or []) if e==FIGHT) for pk in mega] or [0])
    opa=(op.get('active') or [None]); ophp=int(opa[0].get('hp',0)) if (opa and opa[0]) else 0
    ma=(me.get('active') or [None]); mahp=int(ma[0].get('hp',0)) if (ma and ma[0]) else 0
    return [cur.get('turn',0),len(mega),mfe,sum(1 for pk in poks if pk.get('id')==RIOLU),
            sum(len(pk.get('energies') or []) for pk in poks),mahp,ophp,
            len([b for b in (me.get('bench') or []) if b]),len([b for b in (op.get('bench') or []) if b]),
            me.get('handCount',0),me.get('deckCount',0),len(me.get('prize') or []),len(op.get('prize') or [])]
def vev(obs,p):
    cur=obs.get('current') if obs else None
    if cur is None: return -1e9
    r=cur.get('result',-1)
    if r==p: return 1e7
    if r==(1-p): return -1e7
    if r==2: return -1e5
    try: return float(VAL.predict(np.array([feats(cur,p)],dtype=float))[0])
    except Exception: return 0.5
def v_rollout(state,p,cap=80):
    node=state
    for _ in range(cap):
        st=node.get('state',node); obs=st.get('observation'); sid=st.get('searchId')
        cur=obs.get('current') if obs else None
        if cur is None or cur.get('result',-1)!=-1 or cur.get('yourIndex')!=p: return obs
        sel=obs.get('select')
        if sel is None: return obs
        nn=len(sel.get('option') or []); mn=int(sel.get('minCount',1) or 1)
        if nn==0: return obs
        if mn<=1:
            bo,bv,bc=0,-1e18,None
            for oi in range(nn):
                try:
                    ch=raw_step(sid,[oi])
                    if ch.get('error',0)!=0: continue
                    v=vev(ch['state']['observation'],p)
                    if v>bv: bv,bo,bc=v,oi,ch
                except Exception: continue
            if bc is None: return obs
            node=bc
        else:
            try:
                node=raw_step(sid,list(range(min(mn,nn))))
                if node.get('error',0)!=0: return obs
            except Exception: return obs
    return node.get('state',node).get('observation')
def lucario_value_fwdsim(obs_dict,deck):
    sel=obs_dict.get('select')
    if sel is None: return None
    n=len(sel.get('option') or []); mn=int(sel.get('minCount',1) or 1)
    if n<2 or mn>1: return fv(obs_dict)
    try:
        ob=api.to_observation_class(obs_dict); stt=ob.current
        if stt is None or getattr(ob,'search_begin_input',None) is None: return fv(obs_dict)
        p=stt.yourIndex; me=stt.players[p]; opp=stt.players[1-p]
        yd=list(deck); yp=list(deck)[:max(1,len(me.prize))]; od=list(deck)
        op_=list(deck)[:max(1,len(opp.prize))]; oh=list(deck)[:max(1,opp.handCount)]
        oa=[deck[0]] if (len(opp.active)>0 and opp.active[0] is None) else []
        root=api.search_begin(ob,yd,yp,od,op_,oh,oa); rid=root.searchId
        bi,bv=0,-1e18
        for oi in range(n):
            try:
                child=raw_step(rid,[oi])
                end=v_rollout(child['state'],p) if child.get('error',0)==0 else None
                v=vev(end,p) if end else -1e17
            except Exception: v=-1e17
            if v>bv: bv,bi=v,oi
        try: api.search_end()
        except Exception: pass
        return [bi]
    except Exception: return fv(obs_dict)
