import pickle
import pandas as pd
import uproot
from root_pandas import to_root



# load my model
loaded_model = pickle.load(open("Isolation_Classification_Xgboost.dat", "rb"))
model = loaded_model.get_booster().get_dump()
# Convert Model into TMVA-like xml file :
#input_variables = [ (Isoltaion_training_vars , 'F)' ]
Isoltaion_training_vars = [ 'muon_p_Tr_DPHI' , 'muon_p_Tr_MinIPChi2' , 'muon_p_Tr_DETA',
                            'muon_p_Tr_PT'   , 'muon_p_Tr_ANGLE'     ,'muon_p_Tr_DOCA' ,
                            'muon_p_Tr_SVDIS', 'muon_p_Tr_FC_MU'     , 'muon_p_Tr_PVDIS',
                            'muon_p_Tr_DCHI2'
                            ]

#Convert it into list of tuple :
Isoltaion_training_vars_t = [ (x , 'F') for x in Isoltaion_training_vars ]
print(Isoltaion_training_vars_t)


