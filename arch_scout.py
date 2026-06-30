import sys, json
from collections import Counter
sys.path.insert(0,'/kaggle/input/competitions/pokemon-tcg-ai-battle/sample_submission')
from cg.api import all_card_data, all_attack, CardType
CT={c.cardId:c for c in all_card_data()}
ATK={a.attackId:a for a in all_attack()}
deck=json.load(open('/kaggle/working/league_decks.json'))['Archaludon_or_Other']
cnt=Counter(deck)
def atkinfo(c):
    out=[]
    for aid in (getattr(c,'attacks',None) or []):
        a=ATK.get(aid)
        if a: out.append('%s:%sdmg/%sE'%(getattr(a,'name','?')[:14],getattr(a,'damage','?'),len(getattr(a,'energies',None) or [])))
    return out
print('=== Archaludon deck (60) full breakdown ===')
for cid,n in sorted(cnt.items(),key=lambda x:-x[1]):
    c=CT.get(cid)
    if not c: print('  %dx id=%d UNKNOWN'%(n,cid)); continue
    ct=getattr(c,'cardType',None)
    base='  %dx %-24s id=%d type=%s'%(n,getattr(c,'name','?')[:24],cid,str(ct).split('.')[-1])
    if ct==CardType.POKEMON:
        ab=getattr(c,'ability',None) or getattr(c,'abilityName','')
        print(base+' hp=%s wk=%s st1=%s st2=%s ex=%s atk=%s ability=%s'%(
            getattr(c,'hp','?'),getattr(c,'weakness','?'),getattr(c,'stage1',False),getattr(c,'stage2',False),
            getattr(c,'ex',False),atkinfo(c), str(ab)[:30]))
    else:
        print(base+' name_full=%s'%getattr(c,'name','?'))
