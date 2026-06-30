import sys, importlib.util, shutil
shutil.copy('/kaggle/working/v62_main.py','/kaggle/working/agent_v62lucario/main.py')
d={673:2,674:2,675:2,676:3,677:4,678:4,1102:4,1123:2,1141:4,1142:4,1152:2,1159:1,1182:3,1192:4,1227:4,1252:1,6:14}
deck=[]
for k,n in d.items(): deck.extend([k]*n)
assert len(deck)==60, len(deck)
txt='\n'.join(str(c) for c in deck)+'\n'
for f in ['lucario_deck.csv','deck.csv']:
    open('/kaggle/working/agent_v62lucario/'+f,'w').write(txt)
print('deck written', len(deck))
sys.path.insert(0,'/kaggle/input/competitions/pokemon-tcg-ai-battle/sample_submission')
sys.path.insert(0,'/kaggle/working/agent_v62lucario')
sp=importlib.util.spec_from_file_location('v62','/kaggle/working/agent_v62lucario/main.py')
m=importlib.util.module_from_spec(sp); sp.loader.exec_module(m)
print('v62 loads OK, deck len', len(m.agent({'select':None,'logs':[],'current':None})))
