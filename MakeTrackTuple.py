import numpy as np
import pandas as pd
import uproot
from root_pandas import to_root
import pickle

def unnest(frame, explode):
    def mesh(values):
        return np.array(np.meshgrid(*values)).T.reshape(-1, len(values))
    data = np.vstack(mesh(row) for row in frame[explode].values)
    return pd.DataFrame(data=data, columns=explode)
                

def unnesting(df, explode):
    idx = df.index.repeat(df[explode[0]].str.len())
    df1 = pd.concat([
        pd.DataFrame({x: np.concatenate(df[x].values)}) for x in explode],
                    axis=1)
    df1.index = idx
    return df1.join(df.drop(explode, 1), how='left')

#==================================================================
Cuts = "muon_p_P>6000 & muon_p_PT>1500 & kaon_m_IPCHI2_OWNPV>12 & muon_p_IPCHI2_OWNPV>12 & Bs_ENDVERTEX_CHI2<4 & Bs_DIRA_OWNPV>0.999 & kaon_m_P>10000 & kaon_m_PT>800 & Bs_FDCHI2_OWNPV>120"
Isoltaion_training_vars = [ 'muon_p_Tr_DPHI','muon_p_Tr_MinIPChi2','muon_p_Tr_DETA',
                            'muon_p_Tr_PT' , 'muon_p_Tr_ANGLE' , 'muon_p_Tr_DOCA' ,
                            'muon_p_Tr_SVDIS','muon_p_Tr_FC_MU', 'muon_p_Tr_PVDIS',
                            'muon_p_Tr_PAIR_VTCHI2NDOF' , 'muon_p_Tr_MC_ID', 'muon_p_Tr_MATCHCHI2',
                            'muon_p_Tr_MC_MOTHER_ID','muon_p_Tr_MC_GD_MOTHER_ID'
                            ]

mydir = '/eos/lhcb/user/b/bkhanji/ForChristian/AllPVs_NewVars' 
f_Signal = mydir+"/DTT_13512010_Upgrade.root"
f_JpsiK  = mydir+"/DTT_12143001_Upgrade.root"
f_Dsmu   = mydir+"/DTT_13774002_Upgrade.root"
t_name_original = 'Bs2KmuNuTuple/DecayTree'
Signal  = uproot.open(f_Signal)[t_name_original]
JpsiK   = uproot.open(f_JpsiK)[t_name_original]
Dsmu    = uproot.open(f_Dsmu)[t_name_original]

df_Signal = Signal.pandas.df( flatten=False)
df_JpsiK  = JpsiK.pandas.df(flatten=False)
df_Dsmu   = Dsmu.pandas.df(flatten=False)
df_Signal.name = 'Signal'
df_JpsiK.name ='JpsiK'
df_Dsmu.name ='Dsmu'

df_l = [df_Signal,df_JpsiK,df_Dsmu]
for df_i in df_l:
    df_i['muon_p_Tr_DETA']                 = [y for y in df_i['muon_p_Tr_ETA'] ] - df_i['muon_p_ETA']
    df_i['muon_p_Tr_DPHI']                 = [y for y in df_i['muon_p_Tr_PHI'] ] - df_i['muon_p_PHI']
    df_i['log_abs_muon_p_Tr_PVDIS']        = [np.log(np.absolute(y)) for y in df_i['muon_p_Tr_PVDIS'] ]
    df_i['log_abs_muon_p_Tr_SVDIS']        = [np.log(np.absolute(y)) for y in df_i['muon_p_Tr_SVDIS'] ]
    df_i['log_muon_p_Tr_MinIPChi2']        = [np.log(y) for y in df_i['muon_p_Tr_MinIPChi2']]
    df_i['log_muon_p_Tr_PAIR_VTCHI2NDOF']  = [np.log(y) for y in df_i['muon_p_Tr_PAIR_VTCHI2NDOF']]    
    df_i['log_muon_p_Tr_DOCA']             = [np.log(y) for y in df_i['muon_p_Tr_DOCA'] ] 
    df_i_filtered = df_i.query(Cuts)
    df_i_Track = df_i_filtered[Isoltaion_training_vars]
    df_i_Track = unnesting(df_i_Track, Isoltaion_training_vars )
    df_i_Track = df_i_Track.reset_index()
    df_i_Track.to_root(df_i.name+'.root', key='DecayTree')



'''
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
#print(model.predict_proba(df_Signal[ Isoltaion_training_vars ].apply(lambda col: col.str[0]))[:,1] )

#df_Signal['xgboost_IsoBDT'] = model.predict_proba(df_Signal[Isoltaion_training_vars].apply(lambda col: col.str[0]))[:,1]

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
'''
