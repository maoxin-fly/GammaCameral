#!/usr/bin/env python
# coding: utf-8
#=====================================================
#=====================Step one========================
#Combine all binary files
#Calculate Normalized Photon Transport weight martrix
#=====================================================
import os
import struct
import numpy as np

# Read binary files
#===========================================================================
Xsize = 12
Ysize = 12
Zsize = 12
SIPMsize = 6*6*2
data_Sum = np.zeros(shape = Xsize*Ysize*Zsize*SIPMsize,dtype = float)
emission_Sum = 0
print("============Check====================")
print("Merging File:")
for fileNum in range(1,101):
    filename = "data/SimResult_B/data_2020{:.0f}".format(fileNum)
    f = open(file = filename,mode = 'rb')# Read in binary
    size = os.path.getsize(filename)# Calculate bytes
    data_array = np.zeros(shape = int(size/4),dtype = float)
    for i in range(size):
        data = f.read(4)# float = 4 bytes
        if len(data) != 4:
            break
        else:
            data_float = struct.unpack("f",data)[0]# Turn binary to float
            data_array[i] = round(data_float,1)# Approximate
        #print(i,data_array[i])
#     print("data_array:",data_array.shape,data_array)
#     print(data_array[:data_array.shape[0]].sum())
    print(fileNum,end = " ")
#     print("data_Sum:",data_Sum.shape)
     
#     print("indata_size:",size/4)
#     print(6*6*2*12*12*12)

    f.close()
    data_Sum += data_array[0:int(size/4)-1]
    #emission_Sum += data_array[int(size/4)-1]*12*12*12
    emission_Sum += data_array[int(size/4)-1]
    #print(data_array.shape)
    
# Turn 1/8 detector into 100%    
#===========================================================================
data_Sum = data_Sum/(emission_Sum) # Normalize K_is
print(" ")
print(data_Sum.sum(),emission_Sum)
data_Sum.shape = Zsize,Ysize,Xsize,2,6,6  # 1/8 detector
data_Whole_Detector = np.zeros(shape = (23,23,23,2,6,6),dtype = float)
#SIPM and detector symmetry the same time
for z in range(0,23):
    for y in range(0,23):
        for x in range(0,23):
            # z <= 11
            if x <= Xsize - 1 and y <= Ysize - 1 and z <= Zsize - 1:
                data_Whole_Detector[z,y,x,:,:,:] = data_Sum[z,y,x,:,:,:]
            elif x > Xsize - 1 and y <= Ysize - 1 and z <= Zsize - 1:
                data_Whole_Detector[z,y,x,:,:,:] = data_Sum[z,y,22-x,:,:,5::-1]
            elif x > Xsize - 1 and y > Ysize - 1 and z <= Zsize - 1:
                data_Whole_Detector[z,y,x,:,:,:] = data_Sum[z,22-y,22-x,:,5::-1,5::-1]
            elif x <= Xsize - 1 and y > Ysize - 1 and z <= Zsize - 1:
                data_Whole_Detector[z,y,x,:,:,:] = data_Sum[z,22-y,x,:,5::-1,:]
            # z >= 11
            elif x <= Xsize - 1 and y <= Ysize - 1 and z > Zsize - 1:
                data_Whole_Detector[z,y,x,:,:,:] = data_Sum[22-z,y,x,1::-1,:,:]
            elif x > Xsize - 1 and y <= Ysize - 1 and z > Zsize - 1:
                data_Whole_Detector[z,y,x,:,:,:] = data_Sum[22-z,y,22-x,1::-1,:,5::-1]
            elif x > Xsize - 1 and y > Ysize - 1 and z > Zsize - 1:
                data_Whole_Detector[z,y,x,:,:,:] = data_Sum[22-z,22-y,22-x,1::-1,5::-1,5::-1]
            elif x <= Xsize - 1 and y > Ysize - 1 and z > Zsize - 1:
                data_Whole_Detector[z,y,x,:,:,:] = data_Sum[22-z,22-y,x,1::-1,5::-1,:]
print(" ")       
print("============Check====================")
print("data_Sum1/8 shape:",data_Sum.shape)
print("Whole detector shape:",data_Whole_Detector.shape)
print("Emission_Sum:",emission_Sum)

# Normalize
#===========================================================================
Sum_bySIPM = data_Whole_Detector.sum(axis = 5).sum(axis = 4).sum(axis = 3)
data_Whole_Detector_Weight = np.zeros(shape = (23,23,23,2,6,6),dtype = float)
print(Sum_bySIPM.shape)
for z in np.arange(23):
    for y in np.arange(23):
        for x in np.arange(23):
            data_Whole_Detector_Weight[z,y,x,:] = data_Whole_Detector[z,y,x,:]/Sum_bySIPM[z,y,x]



# Save file
#===========================================================================
f1 = open('data/data_Whole_Detector_Weight.bin',mode = 'wb')# Read in binary
#data_Whole_Detector_Weight.reshape(2*6*6*23*23*23).tofile(f1,format = "%f") # binary file, np.float is 64 bytes
# np.savetxt('data/data_Whole_Detector_Weight.txt',data_Whole_Detector_Weight.reshape(6*6*2*23*23*23))  # txt file
np.save('data/data_Whole_Detector_Weight.npy',data_Whole_Detector_Weight,fix_imports=True)
np.save('data/data_Whole_Detector_kis.npy',data_Whole_Detector)
data_Whole_Detector.reshape(2*6*6*23*23*23).tofile('data/data_Whole_Detector_kis.bin',format = "%f") 

print("Whole detector shape weight:",data_Whole_Detector_Weight.shape)
print("Saving data_Whole_Detector_Weight as data/data_Whole_Detector_Weight.npy")
print("Saving data_Whole_Detector as data/data_Whole_Detector_kis.npy")
