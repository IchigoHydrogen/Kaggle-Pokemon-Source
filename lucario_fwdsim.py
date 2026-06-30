"""Pilot = forward-sim + GREEDY rollout with a LUCARIO-AWARE eval + terminal.
greedy-ev is a decent rollout but its eval ignores setup; add a Lucario setup
bonus (Mega Lucario in play + Fighting energy on it = attack-ready) and drop the
irrelevant +5*hand. Deck-specific eval tuning (tractable moat form)."""
import sys
sys.path.insert(0,'/kaggle/working')
import generic_fwdsim_test as G
api=G.api; raw_step=G.raw_step; fv=G.fv; _hp=G._hp
MEGA_LUCARIO=678; RIOLU=677; FIGHTING=6

def luc_ev(obs,p):
    cur=obs.get('current') if obs else None
    if cur is None: return -1e9
    r=cur.get('result',-1)
    if r==p: return 1e7
    if r==(1-p): return -1e7
    if r==2: return -1e5
    me,op=cur['players'][p],cur['players'][1-p]
    mp=len(me.get('prize') or []); opp=len(op.get('prize') or [])
    oa=(op.get('active') or [None]); op_hp=int(oa[0].get('hp',0)) if (oa and oa[0]) else 0
    base=1000.0*(opp-mp)-2.0*op_hp+(_hp(me)-_hp(op))
    # Lucario setup bonus: a Mega Lucario with Fighting energy is attack-ready
    bonus=0.0
    for pk in ([me['active'][0]] if me.get('active') and me['active'][0] else [])+[b for b in (me.get('bench') or []) if b]:
        if pk.get('id')==MEGA_LUCARIO:
            bonus+=120.0                                  # evolved attacker in play
            fe=sum(1 for e in (pk.get('energies') or []) if e==FIGHTING)
            bonus+=60.0*min(fe,2)                         # energy toward Aura Jab(1)/Mega Brave(2)
        elif pk.get('id')==RIOLU:
            bonus+=30.0                                   # development toward Lucario
    return base+bonus

def luc_rollout(state,p,cap=80):
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
            best_oi,best_v,best_child=0,-1e18,None
            for oi in range(nn):
                try:
                    ch=raw_step(sid,[oi])
                    if ch.get('error',0)!=0: continue
                    v=luc_ev(ch['state']['observation'],p)
                    if v>best_v: best_v,best_oi,best_child=v,oi,ch
                except Exception: continue
            if best_child is None: return obs
            node=best_child
        else:
            try:
                node=raw_step(sid,list(range(min(mn,nn))))
                if node.get('error',0)!=0: return obs
            except Exception: return obs
    return node.get('state',node).get('observation')

def lucario_fwdsim(obs_dict,deck):
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
        best_i,best_v=0,-1e18
        for oi in range(n):
            try:
                child=raw_step(rid,[oi])
                end=luc_rollout(child['state'],p) if child.get('error',0)==0 else None
                v=luc_ev(end,p) if end else -1e17
            except Exception: v=-1e17
            if v>best_v: best_v,best_i=v,oi
        try: api.search_end()
        except Exception: pass
        return [best_i]
    except Exception: return fv(obs_dict)
