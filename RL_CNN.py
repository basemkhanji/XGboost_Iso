import numpy as np
import pandas as pd
import uproot
from root_pandas import to_root
import pickle
from matplotlib import pyplot
import tensorflow as tf
from tensorflow import keras
#==================================================================

print(tf.__version__)
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

pd_df_Iso         = dataIso.pandas.df()[:500]
pd_df_NonIsoJpsiK = dataNonIsoJpsiK.pandas.df()[:500]
pd_df_NonIsoDsmu  = dataNonIsoDsmu.pandas.df()[:500]
pd_df_NonIso      = pd.concat([pd_df_NonIsoJpsiK , pd_df_NonIsoDsmu ], ignore_index=True )

Isoltaion_training_vars = [ 'muon_p_Tr_DPHI'  , 'muon_p_Tr_MinIPChi2' , 'muon_p_Tr_DETA',
                            'muon_p_Tr_PT'   , 'muon_p_Tr_ANGLE'     , 'muon_p_Tr_DOCA' ,
                            'muon_p_Tr_SVDIS', 'muon_p_Tr_FC_MU'     , 'muon_p_Tr_PVDIS',
                            ]

#get the tree with training features
train_Iso    = pd_df_Iso[ Isoltaion_training_vars]
train_NonIso = pd_df_NonIso[ Isoltaion_training_vars ]
# #Add column to indicate signal versus backgorund labels for the training :
train_NonIso['Isolating'] = 1
train_Iso['Isolating']    = 0
# add the two data frames
train_data = pd.concat([train_NonIso , train_Iso ], ignore_index=True)
#print train_data.shape[0]
# data frames should be read by tensor flow :
#==================================================================


#==================================================================
# Start the training here :
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
X = train_data[Isoltaion_training_vars]
Y = train_data['Isolating']
# split into test and train datasets
print('Start splitting into tain and test datases')
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size= 0.3, random_state=7)
#X_t = tf.convert_to_tensor( X )
#Y_t = tf.convert_to_tensor( Y )
#x_train  = tf.convert_to_tensor( x_train )
#x_test   = tf.convert_to_tensor( x_test  )
#y_train  = tf.convert_to_tensor( y_train )
#y_test   = tf.convert_to_tensor( y_test  )

'''
model = tf.estimator.DNNClassifier(
    feature_columns= Isoltaion_training_vars ,
    hidden_units= [256, 32] ,
    optimizer=tf.train.AdamOptimizer(1e-4),
    n_classes=10,
    dropout=0.1,
    )
'''
model = keras.Sequential([
    keras.layers.Dense(10, activation=tf.nn.relu , input_shape=(len(Isoltaion_training_vars), )),
    #keras.layers.Dense(10, activation=tf.nn.softmax),
    keras.layers.Dense(4, activation=tf.nn.sigmoid),
    keras.layers.Dense(2 , activation=tf.nn.sigmoid)
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy', #'binary_crossentropy'
              metrics=['accuracy'])
model.fit( x_train , y_train ,epochs = 20)
model.save("test_model.h5")
test_loss, test_acc = model.evaluate( x_test , y_test  )
print('Test accuracy:', test_acc)

predictions = model.predict( X  )
print(predictions[:,0])
#predictions = model.evaluate(feed_dict = {Isoltaion_training_vars:X})
X['Seq_out'] = predictions[:,0]

print(X)
