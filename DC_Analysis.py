#=====================================================
#=====================================================
# Analysis
#=====================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def model(DataSet_A,Pars_mu):
    Process_Num = DataSet_A.shape[0]
    Position_Nums = Pars_mu.shape[0]
    print("Starting griding...........")

    GridID_tem = np.zeros(shape = (Process_Num),dtype = float)
    Pars_mu_minus = np.zeros(shape = (Pars_mu.shape),dtype = float)
    for n in np.arange(Process_Num):
        for i in np.arange(Pars_mu.shape[0]):
            Pars_mu_minus[i,:] = DataSet_A[n,:] - Pars_mu[i,:]
        GridID_tem[n] = np.sum(Pars_mu_minus**2,axis = 1).argmin() # Find best site grid


    Grided_Data = pd.Series(data = GridID_tem)
    GridID_index = np.array(Grided_Data.value_counts().index).astype(int) # IDs for grids has Gammas
    GridID_Distri_P = np.array(Grided_Data.value_counts()) / Process_Num # Normalizing
    Inverted_Distri_P = np.zeros(shape = (Position_Nums),dtype = float)
    Inverted_Distri_P[GridID_index] = GridID_Distri_P # p_i
    print("Process Gamma num is:",Process_Num)
    print("IPD cell counts / Cells =",GridID_index.shape[0],"/",Position_Nums)
    return Inverted_Distri_P

# The Object Function 
# f = f1 + f2 + f3
#----------------------------------------------
def cost(DataSet_A,Real_Distri_P,Inverted_Distri_P,DataSet_A_GridID,Pars_mu):
    ProcessNum = DataSet_A.shape[0]
    sum = np.sum

    Pars_mu_ns = Pars_mu[DataSet_A_GridID.astype(int),:]
    Pars_mu_minus1 = DataSet_A -  Pars_mu_ns
    f1 = (Pars_mu_minus1**2).sum(axis = 1).sum(axis = 0)
    f2 = np.sqrt(ProcessNum)*((Real_Distri_P - Inverted_Distri_P)**2).sum(axis = 0) # Object function 1
    f1 = np.log(f1)
    f2 = np.log(f2)
    return f1,f2,f1+f2

# The Gradient 

Pars_mu_end_2000 = np.load("data/Pars_mu_end_2000_log.npy")
costs = np.load("data/cost_end_2000_log.npy")
Pars_mu_initial = np.load("data/Pars_mu_initial.npy")
Inverted_Distri_P = np.load("data/Inverted_Distri_P.npy")
DataSet_A = np.load('data/data_Real_DataSetA.npy')# A_ns ----> X
DataSet_A_GridID = np.load('data/data_Real_DataSetA_GridID.npy')# A_ns ----> X
Real_Distri_P = np.load("data/data_Real_Site_Distribution.npy")
Real_Pars_mu_is_Photon = np.load("data/data_Real_Pars_mu_is.npy")  # Real w_is from photon simulation




print(costs.shape)
print(costs[:,0] - costs[:,1])
Inverted_Distri_P_end_2000 = model(DataSet_A,Pars_mu_end_2000)
Inverted_Distri_P_Real_Photon_Wis = model(DataSet_A,Real_Pars_mu_is_Photon)

print(Pars_mu_end_2000.shape)
print(costs.shape)

# Calculate real Pars_mu
import pandas as pd
df_DataSet_A = pd.DataFrame(DataSet_A)
df_DataSet_A_Copy = pd.DataFrame(DataSet_A)
df_DataSet_A['GridID'] = DataSet_A_GridID
df_DataSet_A_grouped = df_DataSet_A.groupby(['GridID'],as_index = False).mean()
GridID_index = np.array(df_DataSet_A_grouped['GridID'])
Real_Pars_mu = np.zeros(shape = (Pars_mu_end_2000.shape),dtype = float)
Real_Pars_mu[GridID_index.astype(int),:] = np.array(df_DataSet_A_grouped.iloc[:,1:73])  # a_is ,average
print(Real_Pars_mu.shape)
#print(df_DataSet_A_grouped.iloc[:9])

f1_real,f2_real,cost_real = cost(DataSet_A,Real_Distri_P,Inverted_Distri_P_end_2000,DataSet_A_GridID,Real_Pars_mu)

#Draw costs
fig, ax = plt.subplots(figsize=(12,4))
ax.plot(np.arange(len(costs[:,2])), costs[:,2], 'r')
ax.set_xlabel('Iterations')
ax.set_ylabel('Cost')
plt.title('logf1,f2')
plt.savefig("picture/1_1.png")

#Draw Pars_mu_end
Pars_mu_end1 = Pars_mu_end_2000.copy()
Pars_mu_end1.shape = 2,2,2,2,6,6
y = Pars_mu_end1.reshape(2,2,2,2*6*6)[1,1,1,:]
y1 = Pars_mu_initial.reshape(2,2,2,2*6*6)[1,1,1,:]
y2 = Real_Pars_mu.reshape(2,2,2,2*6*6)[1,1,1,:]
y3 = Real_Pars_mu_is_Photon.reshape(2,2,2,2*6*6)[1,1,1,:]
fig, ax1 = plt.subplots(figsize=(6,4))
ax1.plot(np.arange(len(y)), y,label = 'After Iteration')
ax1.plot(np.arange(len(y1)), y1,label = 'Before Iteration')
ax1.plot(np.arange(len(y2)), y2,label = 'Real DataSetA Average Ais Input')
ax1.plot(np.arange(len(y3)), y3,label = 'Real DataSetA Photon Simulation Input')
plt.legend(loc = 'best')
ax1.set_xlabel('SIPMID')
ax1.set_ylabel('Pars_mu')
plt.title('logf1,f2')
plt.savefig("picture/2_1.png")

# Draw IPD
fig, ax3 = plt.subplots(figsize=(6,4))
ax3.plot(np.arange(len(Real_Distri_P)), Real_Distri_P,label = 'Real_Distri_P')
ax3.plot(np.arange(len(Inverted_Distri_P)), Inverted_Distri_P,label = 'Inverted_Distri_P_Initial')
ax3.plot(np.arange(len(Inverted_Distri_P_end_2000)), Inverted_Distri_P_end_2000,label = 'Inverted_Distri_P_end_2000_Iterations')
ax3.plot(np.arange(len(Inverted_Distri_P_Real_Photon_Wis)), Inverted_Distri_P_Real_Photon_Wis,label = 'Inverted_Distri_P_Real_Photon_Wis')
plt.legend(loc = 'best')
ax3.set_xlabel('GridID')
ax3.set_ylabel('IPD')
plt.title('logf1,f2')
plt.savefig("picture/3_1.png")

# Draw cost function
import math
import numpy as np
fig, ax4 = plt.subplots(figsize=(12,4))
ax4.plot(np.arange(len(costs[:,0])), costs[:,0], ':',label = 'f1')
ax4.plot(np.arange(len(costs[:,1])), costs[:,1], label = 'f2')
#ax4.plot(np.arange(len(costs[:,1])), np.log(costs[:,1]), label = 'ln(f2)')
#ax4.plot(np.arange(len(costs[:,0])), np.log(costs[:,0]),':', label = 'ln(f1)')
ax4.plot(np.arange(len(costs[:,0])), np.zeros(shape = len(costs[:,0])) + f1_real,'_', label = 'f1_real')
ax4.plot(np.arange(len(costs[:,0])), np.zeros(shape = len(costs[:,0])) + f2_real, '_',label = 'f2_real')
#ax4.plot(np.arange(len(costs[:,0])), np.zeros(shape = len(costs[:,0])) + cost_real, ':',label = 'cost_real')
ax4.set_xlabel('Iterations')
ax4.set_ylabel('f1,f2')
plt.legend(loc = 'best')
plt.title('logf1,f2')
plt.savefig("picture/4_1.png")
plt.show()

# Draw real Par_mu


