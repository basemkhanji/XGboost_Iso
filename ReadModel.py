import pickle
from xgboost import XGBClassifier , plot_importance
import uproot
from root_pandas import to_root
import xgboost as xgb
# load my model
bst = xgb.Booster() #init model
model = bst.load_model("Isolation_xgb.bin")
'''
Isoltaion_training_vars = [ 'muon_p_Tr_DPHI'     , 'log_muon_p_Tr_MinIPChi2' , 'muon_p_Tr_DETA',
                            'muon_p_Tr_PT'       , 'muon_p_Tr_ANGLE'         , 'log_muon_p_Tr_DOCA' ,
                            'log_abs_muon_p_Tr_SVDIS',  'muon_p_Tr_FC_MU'         , 'log_abs_muon_p_Tr_PVDIS',
                            'log_muon_p_Tr_PAIR_VTCHI2NDOF'   , 'Isolating'
                            ]

print(model)
mydir = '/eos/lhcb/user/b/bkhanji/ForChristian/AllPVs_NewVars'
f_Signal = mydir+"/DTT_13512010_Upgrade.root"
f_JpsiK  = mydir+"/DTT_12143001_Upgrade.root"
f_Dsmu   = mydir+"/DTT_13774002_Upgrade.root"
t_name   = 'DecayTree'

Signal  = uproot.open(f_Signal)[t_name]
JpsiK   = uproot.open(f_JpsiK )[t_name]
Dsmu    = uproot.open(f_Dsmu  )[t_name]

df_Signal = Signal.pandas.df()
df_JpsiK  = JpsiK.pandas.df()
df_Dsmu   = Dsmu.pandas.df()

df_Signal['xgboost_IsoBDT'] = model.predict_proba(df_Signal[Isoltaion_training_vars])[:,1]
df_Signal.to_root('Signal.root', key='DecayTree')
'''
