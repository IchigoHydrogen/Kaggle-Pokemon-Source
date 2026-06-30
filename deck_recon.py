import sys, json
sys.path.insert(0,'/kaggle/input/competitions/pokemon-tcg-ai-battle/sample_submission')
from cg.api import all_card_data, all_attack, CardType
CT={c.cardId:c for c in all_card_data()}
ATK={a.attackId:a for a in all_attack()}
L=json.load(open('/kaggle/working/league_decks.json'))
def dmg(aid):
    a=ATK.get(aid); 
    return (int(getattr(a,'damage',0) or 0), len(getattr(a,'energies',None) or [])) if a else (0,0)
for dn in ['Hop_Trevenant','Archaludon_or_Other']:
    deck=L[dn]; from collections import Counter
    cnt=Counter(deck)
    print('=== %s (60 cards) ==='%dn)
    poks=[(cid,n) for cid,n in cnt.items() if CT.get(cid) and getattr(CT[cid],'cardType',None)==CardType.POKEMON]
    for cid,n in sorted(poks,key=lambda x:-x[1]):
        c=CT[cid]; atks=getattr(c,'attacks',None) or []
        ad=[(aid,)+dmg(aid) for aid in atks]
        wk=getattr(c,'weakness',None)
        print('  %dx %-22s id=%d hp=%s wk=%s atks(id,dmg,cost)=%s stage1=%s stage2=%s ex=%s'%(
            n, getattr(c,'name','?')[:22], cid, getattr(c,'hp','?'), wk, ad, getattr(c,'stage1',False), getattr(c,'stage2',False), getattr(c,'ex',False)))
