import numpy as np
import pandas as pd
import sys,time
from numba import jit
import matplotlib.pyplot as plt
# Functions
#===========================================================================
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

#Initializing parameter: mu_is 
#----------------------------------------------
@jit
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


# The Predicting Function: Inverted_Distri_P
#----------------------------------------------
@jit
def model(DataSet_A,Pars_mu):
    Process_Num = DataSet_A.shape[0]
    print("Starting griding...........")
    begin_time1_1 = time.time()
    
    GridID_tem = np.zeros(shape = (Process_Num),dtype = float)
    Pars_mu_minus = np.zeros(shape = (Pars_mu.shape),dtype = float)
    for n in np.arange(Process_Num):
        for i in np.arange(Pars_mu.shape[0]):
            Pars_mu_minus[i,:] = DataSet_A[n,:] - Pars_mu[i,:]
        GridID_tem[n] = np.sum(Pars_mu_minus**2,axis = 1).argmin() # Find best site grid
        
    begin_time1_2 = time.time()
    print("Running Time:",begin_time1_2 - begin_time1_1,"s")

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
@jit
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
#----------------------------------------------
@jit
def gradient(DataSet_A, Real_Distri_P, Inverted_Distri_P, Pars_mu, grad_StepSize = 0.001):
    grad = np.zeros(shape = (Pars_mu.shape),dtype = float) # i*s
    for i in range(Pars_mu.shape[0]):
        for s in range(Pars_mu.shape[1]):
    
            Pars_mu_tem_up = Pars_mu.copy()
            Pars_mu_tem_down = Pars_mu.copy()
            Pars_mu_tem_up[i,s] += grad_StepSize
            Pars_mu_tem_down[i,s] -= grad_StepSize
            Pars_mu_tem_up[i,:] / (1 + grad_StepSize)
            Pars_mu_tem_down[i,:] / (1 - grad_StepSize)
            #print(np.sum(Pars_mu_tem_up[i,:]))
            f1_up,f2_up,cost_up = cost(DataSet_A,Real_Distri_P,Inverted_Distri_P,DataSet_A_GridID,Pars_mu_tem_up)
            f1_down,f2_down,cost_down = cost(DataSet_A,Real_Distri_P,Inverted_Distri_P,DataSet_A_GridID,Pars_mu_tem_down)
            #print(i,s)
            grad[i,s] = (cost_up - cost_down) / (2*grad_StepSize)
    return grad
    
#begin_time2_1 = time.time()            
#grads = gradient(DataSet_A, Real_Distri_P, Inverted_Distri_P, Pars_mu, grad_StepSize = 0.001)    #141.9s / Iteration for 6*6*6*1000 39.55s for 6*6*2*1000
#begin_time2_2 = time.time()
#Grad_Abs_Max = np.abs(grads).max()
#print("Running Time:",begin_time2_2 - begin_time2_1,"s")
#print(grads)
#print(Grad_Abs_Max)
    
# Iteration
#----------------------------------------------
@jit
def Iteration(DataSet_A, Real_Distri_P, Inverted_Distri_P, Pars_mu, grad_StepSize = 0.01,Iter_StepSize = 0.001,Iter_Num = 2000):
    init_time = time.time()
    costs = [cost(DataSet_A,Real_Distri_P,Inverted_Distri_P,DataSet_A_GridID,Pars_mu)]   # 2.3s /10000events
    for num in np.arange(Iter_Num):
        #Gradient
        print("Iteration:",num)
        t1 = time.time()
        grads = gradient(DataSet_A, Real_Distri_P, Inverted_Distri_P, Pars_mu, grad_StepSize)
        Grad_Abs_Max = np.abs(grads).max()
        gamma = 0.001 / Grad_Abs_Max
        t2 = time.time()
#         print(grads)
#         print(gamma)
        print("Gradient time:",t2 - t1)
        
        #Iteration
        Pars_mu = Pars_mu - gamma*grads
        t3 = time.time()
        print("Iteration time:",t3 - t2)
        
        # Normalizing
        Sum_SIPM_i = np.sum(Pars_mu,axis = 1)
        for i in range(Pars_mu.shape[0]):
            Pars_mu[i,:] = Pars_mu[i,:] / Sum_SIPM_i[i]
        t4 = time.time()
        print("Normalizing time:",t4 - t3)
        
        Inverted_Distri_P = model(DataSet_A,Pars_mu)

        costs.append(cost(DataSet_A,Real_Distri_P,Inverted_Distri_P,DataSet_A_GridID,Pars_mu))
    return Pars_mu, costs, time.time() - init_time
#===========================================================================




# Calculating Process
#===========================================================================
# <<Crastal Cell Numbers and SIPM Numbers>>
#----------------------------------------------
PositionX_Num = 2
PositionY_Num = 2
PositionZ_Num = 2
PositionNumsXYZ =  np.array([PositionX_Num,PositionY_Num,PositionZ_Num])
Position_Nums = PositionX_Num * PositionY_Num * PositionZ_Num
Channel_Nums = 6*6*2

# Crastal and SIPM cells' centers coordination
#----------------------------------------------
Crastal_X,Crastal_Y,Crastal_Z = Cal_Crastal_Sites(PositionNumsXYZ)
SIPM_X,SIPM_Y,SIPM_Z = Cal_SIPM_Sites()
Coor_Cells = np.array([Crastal_X,Crastal_Y,Crastal_Z])
Coor_SIPMs = np.array([SIPM_X,SIPM_Y,SIPM_Z])
print("==================================================================")
print("---------------------------------1--------------------------------")
print("==================================================================")
print("Calculating Crastal and SIPM cells' centers coordination as:")
print("Coor_Cells[Crastal_X,Crastal_Y,Crastal_Z] and Coor_SIPMs[SIPM_X,SIPM_Y,SIPM_Z]")
print("Crastal coor's shape:",Coor_Cells.shape)
print("SIPM coor's shape:",Coor_SIPMs.shape)
print("Index sorted order: X -> Y -> Z")

# <<Import Data>>
#----------------------------------------------
# Photon simulation results
#data_Whole_Detector_Weight = np.load('data/data_Whole_Detector_Weight.npy') # Nor used
# Groundtrue dataset
DataSet_A = np.load('data/data_Real_DataSetA.npy')# A_ns ----> X
DataSet_A_GridID = np.load('data/data_Real_DataSetA_GridID.npy')# A_ns ----> X
#DataSet_A = np.fromfile('data/data_Real_DataSetA.bin',dtype = np.float64)# A_ns ----> X
#DataSet_A.shape = round(DataSet_A.shape[0] / Channel_Nums), Channel_Nums
#DataSet_A_GridID = np.fromfile('data/data_Real_DataSetA_GridID.bin',dtype = np.float64)# A_ns ----> X
print(DataSet_A.shape,DataSet_A_GridID.shape)
# Real interaction position distribution (IPD)
Real_Distri_P = np.load('data/data_Real_Site_Distribution.npy') # P_i 
# LRF parameter --> weight = mu
Pars_mu = np.zeros(shape = (Position_Nums,Channel_Nums),dtype = float) # mu_is ---->theta
ProcessNum = DataSet_A.shape[0]
print("==================================================================")
print("---------------------------------2--------------------------------")
print("==================================================================")
print("Loading files: w_is(simulation), DataSet_A,DataSet_A_GridID,Real_IPD")
print("shapes:")
print("wis.shape:",Pars_mu.shape)
print("DataSet_A.shape:",DataSet_A.shape)
print("Real_IPD.shape:",Real_Distri_P.shape)
print("DataSet_A_GridID.shape:",DataSet_A_GridID.shape)
begin_time2_1 = time.time()            

# Initializing Parameter mu_is
#----------------------------------------------
Pars_mu = Pars_Initialization(Coor_Cells,Coor_SIPMs)
print("==================================================================")
print("---------------------------------3--------------------------------")
print("==================================================================")
print("Initializing w_is as Pars_mu:")
print(Pars_mu.shape)

# Calculate the Inverted IPD
#----------------------------------------------
Inverted_Distri_P = model(DataSet_A,Pars_mu)
print("==================================================================")
print("---------------------------------4--------------------------------")
print("==================================================================")
print("Iverting IPD as Inverted_Distri_P")
print("Inverted_Distri_P.shape:",Inverted_Distri_P.shape)

# Iteration
#----------------------------------------------
print("==================================================================")
print("---------------------------------5--------------------------------")
print("==================================================================")
print("Start Iterating.......................")
Pars_mu_end, cost_end, time_end = Iteration(DataSet_A, Real_Distri_P, Inverted_Distri_P, Pars_mu, grad_StepSize = 0.001,Iter_StepSize = 0.001,Iter_Num = 2000)

# Saving Results
#----------------------------------------------
np.save('data/Pars_mu_end_2000_log.npy',Pars_mu_end)
np.save('data/cost_end_2000_log.npy',cost_end)
np.save("data/Pars_mu_initial.npy",Pars_mu)
np.save("data/Inverted_Distri_P.npy",Inverted_Distri_P)
#np.save('data/time_end.npy',time_end)
#Pars_mu_end.tofile('data/Pars_mu_end_test.bin',format = "%f")
#np.array(cost_end).tofile('data/cost_end_test.bin',format = "%f")
print(cost_end)
print(time_end,"s")      # 25s/Iteration
begin_time2_1 = time.time()            
