# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
import sys,time
from numba import jit

# Global Quantities:
#===========================================================================
# <<Crastal Cell Numbers and SIPM Numbers>>
#----------------------------------------------
PositionX_Num = 2
PositionY_Num = 2
PositionZ_Num = 2
PositionNumsXYZ =  np.array([PositionX_Num,PositionY_Num,PositionZ_Num])
Position_Nums = PositionX_Num * PositionY_Num * PositionZ_Num
Channel_Nums = 6*6*2
# The Processing Gamma numbers
ProcessNum = 1000


# Calculate Crastal Cell Coordinates: (x_i,y_i.z_i), in mm
#----------------------------------------------
@jit
def Cal_Crastal_Sites(PositionNumsXYZ):
    CellNums = PositionNumsXYZ[0]*PositionNumsXYZ[1]*PositionNumsXYZ[2] #(x,y,z)
    Crastal_SizeXYZ = np.array([20,20,20]) # mm  #(x,y,z)
    
    index = np.arange(CellNums) #Numpy: shape[3] -> shape[2] ->shape[1] -> shape[0]
    index.shape = (PositionNumsXYZ[2],PositionNumsXYZ[1],PositionNumsXYZ[0]) # (NumZ,NumY,NumX) eg (2,6,6)
    X = np.zeros(shape = (CellNums),dtype = float)
    Y = np.zeros(shape = (CellNums),dtype = float)
    Z = np.zeros(shape = (CellNums),dtype = float)
    for i in np.arange(CellNums):
        index_i = np.array(np.where(index == i))
        index_i.shape = (3,)
        #print(index_i)
        X[i] = index_i[2] * (Crastal_SizeXYZ[0] / PositionNumsXYZ[0]) - Crastal_SizeXYZ[0] / 2 + Crastal_SizeXYZ[0] / (2*PositionNumsXYZ[0])
        Y[i] = index_i[1] * (Crastal_SizeXYZ[1] / PositionNumsXYZ[1]) - Crastal_SizeXYZ[1] / 2 + Crastal_SizeXYZ[1] / (2*PositionNumsXYZ[1])
        Z[i] = index_i[0] * (Crastal_SizeXYZ[2] / PositionNumsXYZ[2]) - Crastal_SizeXYZ[2] / 2 + Crastal_SizeXYZ[2] / (2*PositionNumsXYZ[2])
    return X,Y,Z

# Calculate SIPM Coordinates: (x_s,y_s.z_s), in mm
#----------------------------------------------
@jit
def Cal_SIPM_Sites():
    SIPM_gap = 0.29
    SIPM_size = 3.07
    Crastal_size = 20
    Coupling_layer = 0.1
    SIPM_thick = 2
    SIPM_Sites_array_XY = [-2.5*(SIPM_size+SIPM_gap),-1.5*(SIPM_size+SIPM_gap),-0.5*(SIPM_size+SIPM_gap),0.5*(SIPM_size+SIPM_gap),1.5*(SIPM_size+SIPM_gap),2.5*(SIPM_size+SIPM_gap)]
    SIPM_Sites_array_Z = [-(0.5*(Crastal_size + SIPM_thick) + Coupling_layer),0.5*(Crastal_size + SIPM_thick) + Coupling_layer]
    index = np.arange(6*6*2)
    index.shape = (2,6,6) #  (Z,Y,X) X -> Y -> Z 
    X = np.zeros(shape = (6*6*2),dtype = float)
    Y = np.zeros(shape = (6*6*2),dtype = float)
    Z = np.zeros(shape = (6*6*2),dtype = float)
    for s in np.arange(6*6*2):
        index_s = np.array(np.where(index == s))
        index_s.shape = (3,)
        #print(index_s)
        X[s] = SIPM_Sites_array_XY[index_s[2]]
        Y[s] = SIPM_Sites_array_XY[index_s[1]]
        Z[s] = SIPM_Sites_array_Z[index_s[0]]
    return X,Y,Z


Crastal_X,Crastal_Y,Crastal_Z = Cal_Crastal_Sites(PositionNumsXYZ)
SIPM_X,SIPM_Y,SIPM_Z = Cal_SIPM_Sites()
Coor_Cells = np.array([Crastal_X,Crastal_Y,Crastal_Z])
Coor_SIPMs = np.array([SIPM_X,SIPM_Y,SIPM_Z])
print("-----------------------------------------------------------------------------")
print("Calculating Crastal (according to the grid nums) and SIPM cells' centers coordination as:")
print("Coor_Cells[Crastal_X,Crastal_Y,Crastal_Z] and Coor_SIPMs[SIPM_X,SIPM_Y,SIPM_Z]")
print("Crastal coor's shape:",Coor_Cells.shape)
print("SIPM coor's shape:",Coor_SIPMs.shape)
print("Index sorted order: X -> Y -> Z")

#=====================================================
#=====================Step Two========================
# Calculate the <<Real IPD and DataSet_A>>
#=====================================================
#===== 1 ===== Read simulated data   
#===========================================================================
df = pd.read_csv('./data/dataA.txt',header = None,delimiter=" ")
df.columns = ['x','y','z','Edep']
#df.info()
#----------------------------------------------
# Cut an assigned number of events
Gamma_Num = ProcessNum
df_test = df.iloc[0:Gamma_Num].copy()
Light_Yield = 50000 # 1/MeV
Resolution = 0.05  # GAGG crastal
print("Simulated Gamma events:",df.shape[0])
print("Processing Gamma events:",df_test.shape[0])


#===== 2 ===== Calculate photon numbers
#===========================================================================
print("-----------------------------------------------------------------------------")
print('Calculating photon numbers produced by Gamma events...............')
begin_time2_1 = time.time()
@jit
def Cal_Photons(row):
    Mean = Light_Yield * row['Edep']
    PhotonNum = np.round(np.random.normal(Mean,Resolution,1))
    return PhotonNum
df_test['PhotonNums'] = df_test.apply(Cal_Photons,axis = 1)   ### 1.8s/10000 events
df_test['PhotonNums'] = df_test['PhotonNums'].astype(float)
begin_time2_2 = time.time()
print("Running Time:",begin_time2_2 - begin_time2_1,"s")
#print(df_test.iloc[:8])

#===== 3 ===== Calculate real signal site distribution :P_i
#===========================================================================
print("-----------------------------------------------------------------------------")
print('Calculating the real signal site distribution :P_i...............')
begin_time3_1 = time.time()
#----------------------------------------------
# Calculate the grid ID according to the Crastal grid nums
@jit
def Cal_Real_Grid_ID(row):
    Coor_Cells_tem = Coor_Cells.copy()
    for i in np.arange(Coor_Cells.shape[1]):  # Coor_Cells.shape = 3,8
        Coor_Cells_tem[:,i] = Coor_Cells[:,i] - np.array([row['x'],row['y'],row['z']])
    GridID = np.sum(Coor_Cells_tem**2,axis = 0).argmin()
    return GridID

df_test['GridID'] = df_test.apply(Cal_Real_Grid_ID,axis = 1)   ### 56.8s/10000events
df_test['GridID'] = df_test['GridID'].astype(int)
print(df_test.iloc[:5]  )
#----------------------------------------------
# Count grid events and calculate P_i as Distri_P
df_grouped = df_test.groupby('GridID',as_index = False).count() #count
#print(df_grouped.iloc[:10])
GridID_index = np.array(df_grouped['GridID'])
GridID_Distri_P = np.array(df_grouped['x'] / Gamma_Num)
Real_Distri_P = np.zeros(shape = (Position_Nums),dtype = float)
Real_Distri_P[GridID_index] = GridID_Distri_P # p_i
#----------------------------------------------
# Save real site distributions
np.save('data/data_Real_Site_Distribution.npy',Real_Distri_P)
Real_Distri_P.tofile('data/data_Real_Site_Distribution.bin',format = "%f")
print("Saving Real_Distri_P as data/Real_Distri_P.npy")
print("IPD cell counts / Cells =",GridID_index.shape[0],"/",Position_Nums)
begin_time3_2 = time.time()
print("Running Time:",begin_time3_2 - begin_time3_1,"s")

#===== 4 ===== Calculate signal dataset A: a_ns
#===========================================================================
print("-----------------------------------------------------------------------------")
print('Calculating the real signal dataSet :A_ns...............')
begin_time4_1 = time.time()
data_Whole_Detector_kis = np.load('data/data_Whole_Detector_kis.npy')
data_Whole_Detector_kis.shape = (23*23*23,6*6*2) # Only affect the order of the SIPM
data_Whole_Detector_Weight = np.load('data/data_Whole_Detector_Weight.npy')
data_Whole_Detector_Weight.shape = (23*23*23,6*6*2) # Only affect the order of the SIPM
DataSet_A = np.zeros(shape = (Gamma_Num,6*6*2),dtype = float)
DataSet_A_GridID = np.zeros(shape = (Gamma_Num),dtype = float)
# Calculate the cell coordinates of 23*23*23 nums
PositionNumsXYZ_kis =  np.array([23,23,23])
Crastal_X_kis,Crastal_Y_kis,Crastal_Z_kis = Cal_Crastal_Sites(PositionNumsXYZ_kis)
Coor_Cells_kis = np.array([Crastal_X_kis,Crastal_Y_kis,Crastal_Z_kis])
@jit
def Cal_DataSetA(row): # Requiring interpolation from 23*23*23 to different Crastal grid nums
    # Get the appropriate kis
    Photon_kis_num = ((Coor_Cells_kis[0,:] - row['x'])**2 + (Coor_Cells_kis[1,:] - row['y'])**2 + (Coor_Cells_kis[2,:] - row['z'])**2).argmin()
    SIPM_Signal = np.zeros(shape = 6*6*2,dtype = float)
    for s in np.arange(6*6*2):
        mu = row['PhotonNums'] * data_Whole_Detector_kis[Photon_kis_num,s]
        sigma = row['PhotonNums'] * data_Whole_Detector_kis[Photon_kis_num,s] * (1 - data_Whole_Detector_kis[Photon_kis_num,s])
        if sigma == 0:
            SIPM_Signal[s] = mu
        else:
            SIPM_Signal[s] = np.random.normal(mu,sigma,1)
    Sum = np.sum(SIPM_Signal,axis = 0)
    # Normalizing
    if Sum != 0:
        SIPM_Signal = np.abs(SIPM_Signal) / Sum
    else:
        SIPM_Signal = np.abs(SIPM_Signal)
    return SIPM_Signal

def Cal_Real_Pars_mu_is():
    Real_Pars_mu_is = np.zeros(shape = (Coor_Cells.shape[1],6*6*2),dtype = float)
    for i in np.arange(Coor_Cells.shape[1]):
        Photon_kis_num = ((Coor_Cells_kis[0,:] - Coor_Cells[0,i])**2 + (Coor_Cells_kis[1,:] - Coor_Cells[1,i])**2 + (Coor_Cells_kis[2,:] - Coor_Cells[2,i])**2).argmin()
        Real_Pars_mu_is[i,:] = data_Whole_Detector_Weight[Photon_kis_num,:]
    return Real_Pars_mu_is

Real_Pars_mu_is = Cal_Real_Pars_mu_is()
np.save('data/data_Real_Pars_mu_is.npy',Real_Pars_mu_is)


df_test['SIPM_Signal'] = df_test.apply(Cal_DataSetA,axis = 1)    ### 40.2s /10000 events
for n in range(df_test.shape[0]):
    DataSet_A[n,:] = df_test['SIPM_Signal'][n]
    DataSet_A_GridID[n] = df_test['GridID'][n]
print("-----------------------------------------------------------------------------")
begin_time4_2 = time.time()
print("Running Time:",begin_time4_2 - begin_time4_1,"s")
#print(DataSet_A)
#print(DataSet_A_GridID)

np.save('data/data_Real_DataSetA.npy',DataSet_A)
np.save('data/data_Real_DataSetA_GridID.npy',DataSet_A_GridID)
DataSet_A.tofile('data/data_Real_DataSetA.bin',format = "%f")
#DataSet_A_GridID.tofile('data/data_Real_DataSetA_GridID.bin',format = "%f")
print("Saving DataSet_A as data/data_Real_DataSetA.npy")    
print("Saving DataSet_A_GridID as data/data_Real_DataSetA_GridID.npy")

def Pars_Initialization(Coor_Cells,Coor_SIPMs):
    mu_is = np.zeros(shape = (Coor_Cells.shape[1],Coor_SIPMs.shape[1]),dtype = float)
    for i in range(Coor_Cells.shape[1]):
        for s in range(Coor_SIPMs.shape[1]):
            #print(i,s,Coor_Cells[i][0] , Coor_SIPMs[s][0],Coor_Cells[i][1], Coor_SIPMs[s][1],Coor_Cells[i][2] , Coor_SIPMs[s][2])
            mu_is[i][s] = 1. / ((Coor_Cells[0][i] - Coor_SIPMs[0][s])**2 + (Coor_Cells[1][i] - Coor_SIPMs[1][s])**2 + (Coor_Cells[2][i] - Coor_SIPMs[2][s])**2)
    #print(mu_is)
    Sum_SIPM_i = np.sum(mu_is,axis = 1)
    for i in range(Coor_Cells.shape[1]):
        mu_is[i,:] = mu_is[i,:] / Sum_SIPM_i[i]
    return mu_is

# Initializing Parameter mu_is
#----------------------------------------------
Pars_mu = Pars_Initialization(Coor_Cells,Coor_SIPMs)
print("==================================================================")
print("---------------------------------3--------------------------------")
print("==================================================================")
print("Initializing w_is as Pars_mu:")
print(Pars_mu.shape)
print(Pars_mu)
Pars_mu.tofile('data/Initial_Pars_mu_Coor.bin',format = "%f")





