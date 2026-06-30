import sys, importlib.util as u
sys.path.insert(0,'/kaggle/input/competitions/pokemon-tcg-ai-battle/sample_submission')
sys.path.insert(0,'/tmp/agent_v13d4')
s=u.spec_from_file_location('m','/tmp/agent_v13d4/main.py'); m=u.module_from_spec(s); s.loader.exec_module(m)
def mk(opp_active_id, opp_charge, my_hp):
    me={'active':[{'id':m.Alakazam,'hp':my_hp,'energies':[]}],'bench':[],'prize':[1,2,3],'handCount':5}
    op={'active':[{'id':opp_active_id,'hp':100,'energies':[1]*opp_charge}],'bench':[],'prize':[1,2,3],'handCount':5}
    return {'current':{'result':-1,'players':[me,op],'yourIndex':0}}
RACE = lambda: 1000.0*(3-3)-3.0*100  # expected race-eval value = -300
print('expected RACE eval =', RACE())
print('Lucario(Riolu chg1, myHP130) ->', m._rollout_eval(mk(m.Riolu,1,130),0), '(should be -300 = FIRED)')
print('Lucario(Riolu chg0, myHP130) ->', m._rollout_eval(mk(m.Riolu,0,130),0), '(clock=2>1 -> NOT fired)')
print('Alakazam(Abra chg1, myHP130) ->', m._rollout_eval(mk(m.Abra,1,130),0), '(threat90<130 -> NOT fired)')
print('Lucario(Riolu chg1, myHP300)->', m._rollout_eval(mk(m.Riolu,1,300),0), '(270<300 -> NOT fired, my tanky)')
