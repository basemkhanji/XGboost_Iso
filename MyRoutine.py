import numpy as np
import pandas as pd
import uproot
from root_pandas import to_root
import pickle
from xgboost2tmva import convert_model 
#==================================================================
Isoltaion_training_vars = [ 'muon_p_Tr_DPHI','log_muon_p_Tr_MinIPChi2','muon_p_Tr_DETA',
                            'muon_p_Tr_PT' , 'muon_p_Tr_ANGLE' , 'log_muon_p_Tr_DOCA' ,
                            'log_abs_muon_p_Tr_SVDIS','muon_p_Tr_FC_MU', 'log_abs_muon_p_Tr_PVDIS',
                            'log_muon_p_Tr_PAIR_VTCHI2NDOF', 'Isolating' ]
# data manipulation
f_nameIso    = "/afs/cern.ch/user/q/qchristi/public/DTT_13512010_Upgrade_Tracks.root"
f_nameNonIsoJpsiK = "/afs/cern.ch/user/q/qchristi/public/DTT_12143001_Upgrade_Tracks_NIsomu_SIGPV.root"
f_nameNonIsoDsmu  = "/afs/cern.ch/user/q/qchristi/public/DTT_13774002_Upgrade_Tracks_NIsomu_SIGPV.root"
t_name = 'DecayTree'

dataIso          = uproot.open(f_nameIso)[t_name]
dataNonIsoJpsiK  = uproot.open(f_nameNonIsoJpsiK)[t_name]
dataNonIsoDsmu   = uproot.open(f_nameNonIsoDsmu)[t_name]
print('================================================================================')
print(f_nameIso,'file (TTree:',dataIso.name,') has :',dataIso.numentries,'entries \n')
print(f_nameNonIsoJpsiK,'file (TTree:',dataNonIsoJpsiK.name,') has :',dataNonIsoJpsiK.numentries,'entries \n')
print(f_nameNonIsoDsmu,'file (TTree:',dataNonIsoDsmu.name,') has :',dataNonIsoDsmu.numentries, 'entries \n')
print('================================================================================')

#Variable names:
dataIso.keys()
dataIso.arrays(["muon_p_Tr_PT","muon_p_Tr_BPVIP*"] , outputtype=pd.DataFrame , entrystop=10)

pd_df_Iso         = dataIso.pandas.df()#[:500]
pd_df_NonIsoJpsiK = dataNonIsoJpsiK.pandas.df()#[:500]
pd_df_NonIsoDsmu  = dataNonIsoDsmu.pandas.df()#[:500]
pd_df_NonIso      = pd.concat([pd_df_NonIsoJpsiK , pd_df_NonIsoDsmu ], ignore_index=True )

# Add targe and logs to couple of variables:
pd_df_Iso['Isolating']      = 1
pd_df_NonIso['Isolating']   = 0
 
pd_df_Iso['log_abs_muon_p_Tr_PVDIS']  = np.log(np.absolute(pd_df_Iso['muon_p_Tr_PVDIS']) )
pd_df_Iso['log_abs_muon_p_Tr_SVDIS']  = np.log(np.absolute(pd_df_Iso['muon_p_Tr_SVDIS']) )
pd_df_Iso['log_muon_p_Tr_MinIPChi2']  = np.log(pd_df_Iso['muon_p_Tr_MinIPChi2']) 
pd_df_Iso['log_muon_p_Tr_PAIR_VTCHI2NDOF'] = np.log(pd_df_Iso['muon_p_Tr_PAIR_VTCHI2NDOF'])    
pd_df_Iso['log_muon_p_Tr_DOCA']    =  np.log(pd_df_Iso['muon_p_Tr_DOCA'])           

pd_df_NonIso['log_abs_muon_p_Tr_PVDIS']  = np.log(np.absolute(pd_df_Iso['muon_p_Tr_PVDIS']) )
pd_df_NonIso['log_abs_muon_p_Tr_SVDIS']  = np.log(np.absolute(pd_df_Iso['muon_p_Tr_SVDIS']) )
pd_df_NonIso['log_muon_p_Tr_MinIPChi2']  = np.log(pd_df_Iso['muon_p_Tr_MinIPChi2']) 
pd_df_NonIso['log_muon_p_Tr_PAIR_VTCHI2NDOF'] = np.log(pd_df_Iso['muon_p_Tr_PAIR_VTCHI2NDOF'])    
pd_df_NonIso['log_muon_p_Tr_DOCA']    =  np.log(pd_df_Iso['muon_p_Tr_DOCA'])           


#get the tree with training features
train_Iso    = pd_df_Iso[ Isoltaion_training_vars]
train_NonIso = pd_df_NonIso[ Isoltaion_training_vars ]
# #Add column to indicate signal versus backgorund labels for the training :
# add the two data frames
train_data = pd.concat([train_NonIso , train_Iso ], ignore_index=True)
#print train_data.shape[0]
#==================================================================


#==================================================================
# Start the training here :
from xgboost import XGBClassifier , plot_importance
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X = train_data[Isoltaion_training_vars]
Y = train_data['Isolating']
# split into test and train datasets
print('Start splitting into tain and test datases')
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size= 0.3, random_state=7)
print('Start Classification ...')
model = XGBClassifier(objective = "binary:logistic" ,max_depth=5, n_estimators=500, learning_rate=0.05 , nthread=15)
model.fit(x_train, y_train)
print(model)
model.get_booster().dump_model('xgb_model.txt')
import joblib
joblib.dump( model , 'xgb_model.bin') 
model.get_booster().save_model('xgb_model.dat')
from sklearn.metrics import accuracy_score
y_pred = model.predict( x_test ) # Predict using our testdmat
print(accuracy_score(y_pred, y_test))
# append xgboost response back to data :
y_pred_NonIso = model.predict(pd_df_NonIso[Isoltaion_training_vars])
y_pred_Iso    = model.predict(pd_df_Iso[Isoltaion_training_vars])
y_pred_NonIso_proba  = model.predict_proba(pd_df_NonIso[Isoltaion_training_vars])[:,1]
y_pred_Iso_proba     = model.predict_proba(pd_df_Iso[Isoltaion_training_vars])[:,1]
#print(y_pred_NonIso)
#print(y_pred_Iso_proba)

pd_df_Iso['xgboost_IsoBDT']     = y_pred_Iso
pd_df_NonIso['xgboost_IsoBDT']  = y_pred_NonIso
pd_df_Iso['xgboost_IsoBDT_proba']    = y_pred_Iso_proba
pd_df_NonIso['xgboost_IsoBDT_proba'] = y_pred_NonIso_proba
pd_df_NonIso.to_root('NonIso.root', key='DecayTree')
pd_df_Iso.to_root('Iso.root', key='DecayTree')


#print(model.feature_importances_)
from matplotlib import pyplot
plot_importance(model)
#pyplot.show()
pyplot.savefig('./feature_importance.png')
pyplot.close()


mydir = '/eos/lhcb/user/b/bkhanji/ForChristian/AllPVs_NewVars' 
f_Signal = mydir+"/DTT_13512010_Upgrade.root"
f_JpsiK  = mydir+"/DTT_12143001_Upgrade.root"
f_Dsmu   = mydir+"/DTT_13774002_Upgrade.root"
t_name_original = 'Bs2KmuNuTuple/DecayTree'
Signal  = uproot.open(f_Signal)[t_name_original]
#JpsiK   = uproot.open(f_JpsiK )[t_name_original]
#Dsmu    = uproot.open(f_Dsmu  )[t_name_original]

#df_Signal = Signal.pandas.df(['Bs_ENDVERTEX_Y','Bs_ENDVERTEX_Z' , 'muon_p_Tr_MinIPChi2' , 'muon_p_Tr_ETA' ] , entrystop=10, flatten=False)
df_Signal = Signal.pandas.df( flatten=False)[:10]
#df_JpsiK  = JpsiK.pandas.df(flatten=False)
#df_Dsmu   = Dsmu.pandas.df(flatten=False)
Isoltaion_training_vars.remove('Isolating')

#print(df_Signal)
#print(df_Signal[['Bs_ENDVERTEX_Y' , 'Bs_ENDVERTEX_Z']])
#print(df_Signal[ 'muon_p_Tr_MinIPChi2' ])
#print(df_Signal[ ['Bs_ENDVERTEX_Y', 'muon_p_Tr_ETA' ] ] )

#for df_i in [df_Signal , df_JpsiK , df_Dsmu] :
#df_Signal['muon_p_Tr_DETA']                 = np.absolute([y for y in df_Signal['muon_p_Tr_ETA'] ] - df_Signal['muon_p_ETA'])
#df_Signal['muon_p_Tr_DPHI']                 = np.absolute([y for y in df_Signal['muon_p_Tr_PHI'] ] - df_Signal['muon_p_PHI'])
df_Signal['muon_p_Tr_DETA']                 = [y for y in df_Signal['muon_p_Tr_ETA'] ] - df_Signal['muon_p_ETA']
df_Signal['muon_p_Tr_DPHI']                 = [y for y in df_Signal['muon_p_Tr_PHI'] ] - df_Signal['muon_p_PHI']
df_Signal['log_abs_muon_p_Tr_PVDIS']        = [np.log(np.absolute(y)) for y in df_Signal['muon_p_Tr_PVDIS'] ]
df_Signal['log_abs_muon_p_Tr_SVDIS']        = [np.log(np.absolute(y)) for y in df_Signal['muon_p_Tr_SVDIS'] ]
df_Signal['log_muon_p_Tr_MinIPChi2']        = [np.log(y) for y in df_Signal['muon_p_Tr_MinIPChi2']]
df_Signal['log_muon_p_Tr_PAIR_VTCHI2NDOF']  = [np.log(y) for y in df_Signal['muon_p_Tr_PAIR_VTCHI2NDOF']]    
df_Signal['log_muon_p_Tr_DOCA']             = [np.log(y) for y in df_Signal['muon_p_Tr_DOCA'] ] 

#print(df_Signal[ 'muon_p_Tr_DPHI' ])
print(model.predict_proba(df_Signal[ Isoltaion_training_vars ].apply(lambda col: col.str[0]))[:,1] )

df_Signal['xgboost_IsoBDT'] = model.predict_proba(df_Signal[Isoltaion_training_vars].apply(lambda col: col.str[0]))[:,1]

#df_Signal['xgboost_IsoBDT'] = df_Signal[Isoltaion_training_vars].apply(lambda x: model.predict([x])[0],axis=1) #)[:,1]
#print( [model.predict(y)[0] for x,y in df_Signal[ [Isoltaion_training_vars] ].iterrows() ] )
#print( df_Signal[ Isoltaion_training_vars ].loc() )
#print(  model.predict_proba(df_Signal[Isoltaion_training_vars])[:,1])
#print( [  x for [x] in df_Signal[Isoltaion_training_vars] ] )
#df_JpsiK['xgboost_IsoBDT'] = model.predict_proba(df_JpsiK[Isoltaion_training_vars])[:,1]
#df_Dsmu['xgboost_IsoBDT'] = model.predict_proba(df_Dsmu[Isoltaion_training_vars])[:,1]
#print(df_Signal)
#df_Signal.to_root('Signal.root', key='DecayTree')
#df_JpsiK.to_root('JpsiK.root', key='DecayTree')
#df_Dsmu.to_root('Dsmu.root', key='DecayTree')
# put back the xgboost model into the original canddiate-based tuples:


#OR
#pyplot.bar(range(len(model.feature_importances_)), model.feature_importances_)
#pyplot.show()

# Convert Model into TMVA-like xml file :
#input_variables = [ (Isoltaion_training_vars , 'F)' ]
#convert_model(model, input_variables=[('var1','F'),('var2','I')],output_xml='xgboost.xml')
