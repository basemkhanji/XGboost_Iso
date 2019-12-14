import ROOT , sys , os , threading ,subprocess, multiprocessing , time
from ROOT import *
import numpy as np

mydir  = "/eos/lhcb/user/b/bkhanji/ForChristian/AllPVs/"
Dsmunu  = TFile( mydir + "DTT_13774002_Py8_Upgrade.root" , "OPEN")
mychain_dsmunu  = Dsmunu.Get("Bs2KmuNuTuple/DecayTree");
JpsiK  = TFile( mydir + "DTT_12143001_Py8_Upgrade.root" , "OPEN")
mychain_jpsik  = JpsiK.Get("Bs2KmuNuTuple/DecayTree");
Signal  = TFile( mydir + "DTT_13512010_Py8_Upgrade.root" , "OPEN")
mychain_Signal  = Signal.Get("Bs2KmuNuTuple/DecayTree");

#mydir_CQ   = "/afs/cern.ch/user/q/qchristi/public/"
mydir_CQ   = "/eos/lhcb/user/b/bkhanji/ForChristian/AllPVs_NewVars/" 
JpsiK_CQ  = TFile( mydir_CQ + "DTT_12143001_Upgrade_upgradeiso.root" , "OPEN")
mychain_jpsik_CQ  = JpsiK_CQ.Get("DecayTree");
Dsmu_CQ  = TFile( mydir_CQ + "DTT_13774002_Upgrade_upgradeiso.root" , "OPEN")
mychain_dsmu_CQ  = Dsmu_CQ.Get("DecayTree");
Signal_CQ  = TFile( mydir_CQ + "DTT_13512010_Upgrade_upgradeiso.root" , "OPEN")
mychain_Signal_CQ  = Signal_CQ.Get("DecayTree");

Cuts ="muon_p_P>6"#000 & muon_p_PT>1500 & kaon_m_IPCHI2_OWNPV>12 & muon_p_IPCHI2_OWNPV>12 & Bs_ENDVERTEX_CHI2<4 & Bs_DIRA_OWNPV>0.999 & kaon_m_P>10000 & kaon_m_PT>800 & Bs_FDCHI2_OWNPV>120"

c0 = TCanvas()
mychain_dsmunu.Draw("muon_p_IsoMinBDT" , "muon_p_IsoMinBDT<-0.7" , "norm")
mychain_Signal.Draw("muon_p_IsoMinBDT" , "muon_p_IsoMinBDT<-0.7" , "normsame")
mychain_jpsik.Draw("muon_p_IsoMinBDT" , "muon_p_IsoMinBDT<-0.7" , "normsame")

c1 = TCanvas()
mychain_dsmu_CQ.Draw("MinIsoBDT_upgrade" , "MinIsoBDT_upgrade<-0.7" , "norm")
mychain_Signal_CQ.Draw("MinIsoBDT_upgrade" , "MinIsoBDT_upgrade<-0.7" , "normsame")
mychain_jpsik_CQ.Draw( "MinIsoBDT_upgrade" , "MinIsoBDT_upgrade<-0.7" , "normsame")


my_Sigeff  = []
my_bkgRej_jpsik  = []
my_bkgRej_dsmu  = []
my_Sigeff_CQ  = []
my_bkgRej_jpsik_CQ  = []
my_bkgRej_dsmu_CQ  = []

for i in np.linspace( -1., 1. , 100 ):
    eff_i = float(mychain_Signal.GetEntries(Cuts + " && Bs_TruthMatched==1 && muon_p_IsoMinBDT>"+str( i )))/mychain_Signal.GetEntries(Cuts + " && Bs_TruthMatched==1")
    my_Sigeff.append( eff_i )

    rej_i_j = 1-float(mychain_jpsik.GetEntries(Cuts+" && muon_p_IsoMinBDT>"+str( i )))/mychain_jpsik.GetEntries(Cuts)
    my_bkgRej_jpsik.append( rej_i_j )
    rej_i_d = 1-float(mychain_dsmunu.GetEntries(Cuts+" && muon_p_IsoMinBDT>"+str( i )))/mychain_dsmunu.GetEntries(Cuts)
    my_bkgRej_dsmu.append( rej_i_d )
    # ----
    eff_i_CQ = float(mychain_Signal_CQ.GetEntries(Cuts+"&& Bs_TruthMatched==1 &&MinIsoBDT_upgrade>"+str( i )))/mychain_Signal_CQ.GetEntries(Cuts+ " && Bs_TruthMatched==1")
    my_Sigeff_CQ.append( eff_i_CQ )

    rej_i_j_CQ = 1-float(mychain_jpsik_CQ.GetEntries(Cuts+"&& MinIsoBDT_upgrade>"+str( i )))/mychain_jpsik_CQ.GetEntries(Cuts)
    my_bkgRej_jpsik_CQ.append( rej_i_j_CQ )
    rej_i_d_CQ = 1-float(mychain_dsmu_CQ.GetEntries(Cuts+"&& MinIsoBDT_upgrade>"+str( i )))/mychain_dsmu_CQ.GetEntries(Cuts)
    my_bkgRej_dsmu_CQ.append( rej_i_d_CQ )
    

print my_Sigeff
print my_bkgRej_jpsik

#print my_bkgRej_dsmu
#print my_Sigeff_CQ
#print my_bkgRej_jpsik_CQ
#print my_bkgRej_dsmu_CQ


import matplotlib.pyplot as plt

plt.plot(my_bkgRej_jpsik, my_Sigeff , "r" , label = 'RunI')
plt.plot(my_bkgRej_jpsik_CQ ,my_Sigeff_CQ , "g" , label = 'Upgrade')
plt.xlabel('Signal Eff.')
plt.ylabel('Backgorund Rej.')
plt.legend()#handles=[blue_line])
plt.show()
'''

plt.plot(my_bkgRej_dsmu, my_Sigeff , "r" , label = 'RunI')
plt.plot(my_bkgRej_dsmu_CQ ,my_Sigeff_CQ , "g" , label = 'Upgrade')
plt.xlabel('Signal Eff.')
plt.ylabel('Backgorund Rej.')
plt.legend()#handles=[blue_line])
plt.show()
'''


