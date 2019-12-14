#!/usr/bin/env python
import ROOT , sys , os , threading ,subprocess, multiprocessing , time
from ROOT import *
from math import *
from array import array
from multiprocessing import Pool
from os.path import isfile

#InputFile     = TFile( "/afs/cern.ch/user/q/qchristi/public/DTT_13512010_Upgrade_Tracks.root" , "OPEN")
#InputFile     = TFile( "/eos/lhcb/user/b/bkhanji/ForChristian/AllPVs_NewVars/DTT_13512010_Upgrade.root" , "OPEN")
InputFile     = TFile( "/eos/lhcb/user/b/bkhanji/ForChristian/AllPVs_NewVars/DTT_12143001_Upgrade.root" , "OPEN")
#InputFile     = TFile( "/eos/lhcb/user/b/bkhanji/ForChristian/AllPVs_NewVars/DTT_13774002_Upgrade.root" , "OPEN")
mychain       = InputFile.Get("Bs2KmuNuTuple/DecayTree");

#outfile = TFile( "/eos/lhcb/user/b/bkhanji/ForChristian/AllPVs_NewVars/DTT_13512010_Upgrade_upgradeiso.root" , "RECREATE");
outfile = TFile( "/eos/lhcb/user/b/bkhanji/ForChristian/AllPVs_NewVars/DTT_12143001_Upgrade_upgradeiso.root" , "RECREATE");
#outfile = TFile( "/eos/lhcb/user/b/bkhanji/ForChristian/AllPVs_NewVars/DTT_13774002_Upgrade_upgradeiso.root" , "RECREATE");
outfile.cd();

BDTreader         =  TMVA.Reader("!Color:!Silent")
log_muon_p_Tr_MinIPChi2 =array('f',[0])
muon_p_Tr_DETA       =array('f',[0])
muon_p_Tr_DPHI       =array('f',[0])
muon_p_Tr_PT      =array('f',[0])
muon_p_Tr_MATCHCHI2=array('f',[0])
log_abs_muon_p_Tr_PVDIS   =array('f',[0])
log_abs_muon_p_Tr_SVDIS   =array('f',[0])
log_muon_p_Tr_DOCA    =array('f',[0])
muon_p_Tr_ANGLE   =array('f',[0])
muon_p_Tr_FC_MU   =array('f',[0])

BDTreader.AddVariable( "log(muon_p_Tr_MinIPChi2)", log_muon_p_Tr_MinIPChi2 );
BDTreader.AddVariable( "muon_p_Tr_DETA", muon_p_Tr_DETA  );
BDTreader.AddVariable( "muon_p_Tr_DPHI", muon_p_Tr_DPHI );
BDTreader.AddVariable( "log(muon_p_Tr_PT)", muon_p_Tr_PT );
BDTreader.AddVariable( "muon_p_Tr_MATCHCHI2", muon_p_Tr_MATCHCHI2 );
BDTreader.AddVariable( "log(abs(muon_p_Tr_PVDIS))", log_abs_muon_p_Tr_PVDIS );
BDTreader.AddVariable( "log(abs(muon_p_Tr_SVDIS))", log_abs_muon_p_Tr_SVDIS );
BDTreader.AddVariable( "log(muon_p_Tr_DOCA)", log_muon_p_Tr_DOCA );
BDTreader.AddVariable( "muon_p_Tr_ANGLE", muon_p_Tr_ANGLE );
BDTreader.AddVariable( "muon_p_Tr_FC_MU", muon_p_Tr_FC_MU );
                               
#BDTreader.BookMVA('BDTG','./TMVAClassification_BDTG.weights.xml')
BDTreader.BookMVA('BDTG','./TMVAClassification_BDTG.weights.xml')
MyBDTTree = mychain.CopyTree('0')
BDTOutput = array( 'f', [ 0 ] )
MinIsoBDT_upgrade  = array( 'f', [ 0 ] )
#MinXGBOOST_upgrade = array( 'f', [ 0 ] )
#import pickle
#loaded_model = pickle.load(open("xgb_model.txt", "rb"))
#MyBDTTree.Branch('BDTOutput', BDTOutput,'BDTOutput/F')
MyBDTTree.Branch('MinIsoBDT_upgrade', MinIsoBDT_upgrade,'MinIsoBDT_upgrade/F')
#MyBDTTree.Branch('MinXGBOOST_upgrade', MinXGBOOST_upgrade , 'MinXGBOOST_upgrade/F')
#print "N =" ,  MyBDTTree.GetEntries()
n = 0 
for event in mychain:
    #if n > 200 : continue
    #print n
    #print "Bs PT " , mychain.Bs_PT
    a = 0
    xgboost_v_min         = -1
    MinIsoBDT_upgrade_min = 1.5
    #print "==============================================================="
    #print event.NTr
    for ntrack in range(event.NTr):
        if(mychain.muon_p_Tr_DOCA[ntrack]<=0): continue
        log_muon_p_Tr_MinIPChi2[0] = log(mychain.muon_p_Tr_MinIPChi2[ntrack])
        muon_p_Tr_DETA[0]          = mychain.muon_p_Tr_ETA[ntrack] - mychain.muon_p_ETA    
        muon_p_Tr_DPHI[0]          = mychain.muon_p_Tr_PHI[ntrack] - mychain.muon_p_PHI
        muon_p_Tr_PT[0]            = log(mychain.muon_p_Tr_PT[ntrack])
        muon_p_Tr_MATCHCHI2[0]     = mychain.muon_p_Tr_MATCHCHI2[ntrack]
        log_abs_muon_p_Tr_PVDIS[0] = log(abs(mychain.muon_p_Tr_PVDIS[ntrack]))  
        log_abs_muon_p_Tr_SVDIS[0] = log(abs(mychain.muon_p_Tr_SVDIS[ntrack]))
        log_muon_p_Tr_DOCA[0]      = log(mychain.muon_p_Tr_DOCA[ntrack])
        muon_p_Tr_ANGLE[0]         = mychain.muon_p_Tr_ANGLE[ntrack]   
        muon_p_Tr_FC_MU[0]         = mychain.muon_p_Tr_FC_MU[ntrack]
        
        BDTOutput[0]         = BDTreader.EvaluateMVA('BDTG')
        #print ntrack , " " , log_muon_p_Tr_MinIP,muon_p_Tr_DETA,muon_p_Tr_DPHI,log_muon_p_Tr_PT
        #BDTOutput                    = BDTreader.EvaluateMVA('BDTG')
        #print " My BDT value is : " , BDTOutput
        
        if ( BDTOutput[0] < MinIsoBDT_upgrade_min):
            
            MinIsoBDT_upgrade_min = BDTOutput[0]
            #print 'my new min is : ' , MinIsoBDT_upgrade_min
        # write xgbosst prediction
        #'muon_p_Tr_DPHI'     , 'log_muon_p_Tr_MinIPChi2' , 'muon_p_Tr_DETA',
        #'muon_p_Tr_PT'       , 'muon_p_Tr_ANGLE'         , 'log_muon_p_Tr_DOCA' ,
        #'log_abs_muon_p_Tr_SVDIS',  'muon_p_Tr_FC_MU'         , 'log_abs_muon_p_Tr_PVDIS',
        #'log_muon_p_Tr_PAIR_VTCHI2NDOF'   
        #Feature_vector = [muon_p_Tr_DPHI[0]  ,log_muon_p_Tr_MinIP[0],muon_p_Tr_DETA[0],
        #                  log_muon_p_Tr_PT[0],muon_p_Tr_ANGLE[0],log_muon_p_Tr_DOCA[0],
        #                  log_abs_muon_p_Tr_SVDIS[0],muon_p_Tr_FC_MU[0],log_abs_muon_p_Tr_PVDIS[0],
        #                  log_muon_p_Tr_PAIR_VTCHI2NDOF[0]
        #                  ]
        #Feature_vector = np.array(Feature_vector).reshape((1,-1))
        #xgboost_v = loaded_model.predict( Feature_vector)[0]
        #print 'xgboost_v : ' , xgboost_v
        #if(xgboost_v > xgboost_v_min):
        #    xgboost_v_min = xgboost_v
        MinIsoBDT_upgrade[0]  = MinIsoBDT_upgrade_min    
        MyBDTTree.Fill()
    #print '===================================================='
    #print MinIsoBDT_upgrade[0]
    
    #MinXGBOOST_upgrade[0] = xgboost_v_min
    
    n = n+1

    
MyBDTTree.Write("", TObject.kOverwrite)
outfile.Close()
InputFile.Close()
