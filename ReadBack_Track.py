#!/usr/bin/env python
import ROOT , sys , os , threading ,subprocess, multiprocessing , time
from ROOT import *
from math import *
from array import array
from multiprocessing import Pool
from os.path import isfile



#InputFile     = TFile( "/afs/cern.ch/user/q/qchristi/public/DTT_13512010_Upgrade_Tracks.root" , "OPEN")
InputFile     = TFile( "/afs/cern.ch/user/q/qchristi/public/DTT_13774002_Upgrade_Tracks_NIsomu_SIGPV.root" , "OPEN")
#InputFile     = TFile( "/eos/lhcb/user/b/bkhanji/ForChristian/AllPVs_NewVars/DTT_13512010_Upgrade.root" , "OPEN")
#InputFile     = TFile( "/eos/lhcb/user/b/bkhanji/ForChristian/AllPVs_NewVars/DTT_12143001_Upgrade.root" , "OPEN")
mychain       = InputFile.Get("DecayTree");

#outfile = TFile( "/eos/lhcb/user/b/bkhanji/ForChristian/AllPVs_NewVars/DTT_13512010_Upgrade_withnewiso.root" , "RECREATE");
outfile = TFile( "/eos/lhcb/user/b/bkhanji/ForChristian/AllPVs_NewVars/DTT_13774002_Upgrade_withnewiso.root" , "RECREATE");
outfile.cd();

BDTreader         =  TMVA.Reader("!Color:!Silent")
log_muon_p_Tr_MinIP   =array('f',[0])
muon_p_Tr_DETA    =array('f',[0])
muon_p_Tr_DPHI    =array('f',[0])
log_muon_p_Tr_PT      =array('f',[0])
log_muon_p_Tr_PAIR_VTCHI2NDOF=array('f',[0])
log_abs_muon_p_Tr_PVDIS   =array('f',[0])
log_abs_muon_p_Tr_SVDIS   =array('f',[0])
log_muon_p_Tr_DOCA    =array('f',[0])
muon_p_Tr_ANGLE   =array('f',[0])
muon_p_Tr_FC_MU   =array('f',[0])

BDTreader.AddVariable( "log(muon_p_Tr_MinIPChi2)", log_muon_p_Tr_MinIP );
BDTreader.AddVariable( "muon_p_Tr_DETA", muon_p_Tr_DETA  );
BDTreader.AddVariable( "muon_p_Tr_DPHI", muon_p_Tr_DPHI );
BDTreader.AddVariable( "log(muon_p_Tr_PT)", log_muon_p_Tr_PT );
BDTreader.AddVariable( "log(muon_p_Tr_PAIR_VTCHI2NDOF)", log_muon_p_Tr_PAIR_VTCHI2NDOF );
BDTreader.AddVariable( "log(abs(muon_p_Tr_PVDIS))", log_abs_muon_p_Tr_PVDIS );
BDTreader.AddVariable( "log(abs(muon_p_Tr_SVDIS))", log_abs_muon_p_Tr_SVDIS );
BDTreader.AddVariable( "log(muon_p_Tr_DOCA)", log_muon_p_Tr_DOCA );
BDTreader.AddVariable( "muon_p_Tr_ANGLE", muon_p_Tr_ANGLE );
BDTreader.AddVariable( "muon_p_Tr_FC_MU", muon_p_Tr_FC_MU );
                               
BDTreader.BookMVA('BDTG','./TMVAClassification_BDTG.weights.xml')
MyBDTTree = mychain.CopyTree('0')
BDTOutput = array( 'f', [ 0 ] )
MinIsoBDT_upgrade = array( 'f', [ 0 ] )

MyBDTTree.Branch('BDTOutput', BDTOutput,'BDTOutput/F')
MyBDTTree.Branch('MinIsoBDT_upgrade', MinIsoBDT_upgrade,'MinIsoBDT_upgrade/F')

print "N =" ,  MyBDTTree.GetEntries()
n = 0 
for event in mychain:
    if n > 1000 : continue
    print n
    #print "Bs PT " , mychain.Bs_PT
    a = 0
    log_muon_p_Tr_MinIP[0]        = log(mychain.muon_p_Tr_MinIP)
    muon_p_Tr_DETA[0]             = mychain.muon_p_Tr_DETA    
    muon_p_Tr_DPHI[0]             = mychain.muon_p_Tr_DPHI
    log_muon_p_Tr_PT[0]           = log(mychain.muon_p_Tr_PT) 
    log_muon_p_Tr_PAIR_VTCHI2NDOF[0] = log(mychain.muon_p_Tr_PAIR_VTCHI2NDOF)
    log_abs_muon_p_Tr_PVDIS[0]    = log(abs(mychain.muon_p_Tr_PVDIS ))  
    log_abs_muon_p_Tr_SVDIS[0]    = log(abs(mychain.muon_p_Tr_SVDIS ))  
    log_muon_p_Tr_DOCA[0]         = log(mychain.muon_p_Tr_DOCA)
    muon_p_Tr_ANGLE[0]            = mychain.muon_p_Tr_ANGLE   
    muon_p_Tr_FC_MU[0]            = mychain.muon_p_Tr_FC_MU
    BDTOutput[0]                  = BDTreader.EvaluateMVA('BDTG')
    MyBDTTree.Fill()
    
    n = n+1

    
MyBDTTree.Write("", TObject.kOverwrite)
outfile.Close()
InputFile.Close()
4
