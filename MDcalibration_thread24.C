//##########################################################
//注意最后一个线程的循环数目到EventNum;
//##########################################################

#ifndef __CINT_
#include <iostream>
#include <cmath>
#include <time.h>
#include<stdlib.h>
#include<string>
#include "string.h"
#include<stdio.h>
#include <future>
#include <cmath>
#include <mutex>
#include <vector>

using namespace std;

//===========================================================================
//DECLARING
//===========================================================================
//Declare Global Variables
int EventNum;
//-----------------------------------------------------
const int PositionNum_x = 2;
const int PositionNum_y = 2;
const int PositionNum_z = 2;
const int PositionNum = PositionNum_x * PositionNum_y * PositionNum_z;
//-----------------------------------------------------
const int ChannelNum = 2*6*6;
//-----------------------------------------------------
float weight[PositionNum][ChannelNum];//w_is
float Distri_real[PositionNum];//p_i
//-----------------------------------------------------
float StepSize_dif = 0.001;//Stepsize to calculate gradient
float StepSize_cal = 0.0001;//Stepsize to iteration
int* BMUID;//Inverted belonging cell_ID for gamma events
float** BMUDistance;  //Collect S[n][i]

//Declare Structor: to save MinDistanceID && MinDistance
struct BMU 
{
	unsigned int MinDistanceID;
	float MinDistance;
	float Distances[PositionNum];
};

//Declare Functions: Griding function && Cost function
BMU CalBMU(float signal[ChannelNum], float weight[PositionNum][ChannelNum], int BMUID_old);
BMU CalBMU_Gradient(float signal[ChannelNum], float weight_value[PositionNum][ChannelNum],float distances[PositionNum], int GradientID);
float objfunc(float** DataSet, float weight_value[PositionNum][ChannelNum],int GradientID);


//===========================================================================
//MAIN FUNCTION
//===========================================================================
int main(int argc, char* argv[])
{
//GroundTrue: DataSetA[N][S]
//================================================================	
	//Read from binary file -> array
	int GammaNum = atoi(argv[1]);
	int IterNum = atoi(argv[2]);

	char fDataSet[255],fRealIPD[255];
	sprintf(fDataSet,"data/Result_Gamma/Gamma_data_DataSetA_%d.bin",GammaNum);
	sprintf(fRealIPD,"data/Result_Gamma/Gamma_data_Real_IPD_%d.bin",GammaNum);
	FILE* fp = fopen(fDataSet, "rb");//64bytes, numpy_float = 64 bytes
	fseek(fp, 0L, SEEK_END);//指针定位到结尾
	EventNum = ftell(fp) / sizeof(float) / ChannelNum;//算出总事例数
	fseek(fp, 0L, SEEK_SET);//定位到开头
	float** DataSetA = (float**)malloc(sizeof(float*) * EventNum);//分配总事例数I内存空间
	DataSetA[0] = (float*)malloc(sizeof(float) * EventNum * ChannelNum);//分配总信号IS的内存空间
	for(int n=1; n<EventNum; n++)
	{
		DataSetA[n] = DataSetA[n-1] + ChannelNum;//指定数组长度
	}
	fread(DataSetA[0], sizeof(float), ChannelNum * EventNum, fp);//将ChannelNum * EventNum个float长度的数据读进指针
	fclose(fp);
	
	BMUID = (int*)malloc(EventNum * sizeof(int));//Declare the cellID where gamma event belong to
	BMUDistance = (float**)malloc(sizeof(float*) * EventNum);
	BMUDistance[0] = (float*)malloc(sizeof(float) * EventNum * PositionNum);//N*I
	for(int n=1; n<EventNum; n++)
	{
		BMUDistance[n] = BMUDistance[n-1] + PositionNum;//指定数组长度
	}
	
	cout<<"GammaNum:"<<" "<<EventNum<<endl;

//Initial Parameter: weight[I][S] 
//================================================================
	FILE* fp2 = fopen("data/Result_Gamma/Gamma_data_Initial_Pars_mu.bin", "rb");//64 bytes
	fread(weight, sizeof(float), sizeof(weight)/sizeof(float), fp2);//读进weight2
	fclose(fp2);
	const int I = sizeof(weight) / (sizeof(weight[0][0])*72);//Cell nums for crastal
	const int S = sizeof(weight[0]) / sizeof(weight[0][0]);//Channel nums for SiPM
	float sum[I]={0};

//Real IPD: Distri_real[i] (P_i)
// ================================================================
	FILE* fp_distri = fopen(fRealIPD, "rb");
	fread(Distri_real, sizeof(float), sizeof(Distri_real)/sizeof(float), fp_distri);
	fclose(fp_distri);
	float sum_P = 0;

//Start Iteration
// ================================================================
	int IterID = 0;
	char fCost[255],fPars_mu[255];
	sprintf(fCost,"data/Results_Iteration/CPlus_Cost_function_value_%d_%d.bin",GammaNum,IterNum);
	sprintf(fPars_mu,"data/Results_Iteration/CPlus_Pars_mu_end_%d_%d.bin",GammaNum,IterNum);
	//sprintf(fCost,"data/Results_Iteration/Test_f2_CPlus_Cost_function_value_%d_%d.bin",GammaNum,IterNum);
	//sprintf(fPars_mu,"data/Results_Iteration/Test_f2_CPlus_Pars_mu_end_%d_%d.bin",GammaNum,IterNum);
	FILE* fp3 = fopen(fCost, "wb");
	FILE* fp4 = fopen(fPars_mu, "wb");
	int t_start = clock();
	while(IterID<IterNum)
	{
		//-----------------------------------------------------
		//(1).Griding the Gamma Events based on Pars_weight
		for(int n = 0; n<EventNum; n++)
		{
			BMU a = CalBMU(DataSetA[n], weight, -1);//"-1" for looping the whole crastal at first time
			BMUID[n] = a.MinDistanceID;//The belonged cellID
			for(int i = 0;i<PositionNum;i++){
				BMUDistance[n][i] = a.Distances[i];
			}
		}
		int t1 = clock();
		//-----------------------------------------------------
		//(2).Calculate & save the cost function value
		float objfunc_old = objfunc(DataSetA, weight,-1);//Cost function value
		cout.precision(20);
		cout<<"Current iteration number: "<<IterID<<endl;
		cout<<"Current object function value: "<<objfunc_old<<endl;
		float IterID_float = float(IterID);
		fwrite(&IterID_float, sizeof(float), 1,fp3);//Save
		fwrite(&objfunc_old, sizeof(float), 1,fp3);
		//-----------------------------------------------------
		//(3).Calculate the cost function gradient
		//[f(x+delta)-f(x-delta)]/(2*delta)
		float weight_up[PositionNum][ChannelNum];
		float weight_down[PositionNum][ChannelNum];
 		float objchange_up[PositionNum][ChannelNum];
 		float objchange_down[PositionNum][ChannelNum];
		float objchange[PositionNum][ChannelNum];
		
		for(int i = 0; i < PositionNum; i++)//I
		{
			for(int s = 0; s < ChannelNum; s++)//S
			{
				//int t1 = clock();
				memcpy(weight_up, weight, sizeof(weight));//copy the array
				memcpy(weight_down, weight, sizeof(weight));
				weight_up[i][s] = weight_up[i][s] + StepSize_dif;
				weight_down[i][s] = weight_down[i][s] - StepSize_dif;
				for(int w = 0; w < ChannelNum; w++)
				{
					weight_up[i][w] = weight_up[i][w] / (1 + StepSize_dif);//Normalizing weight_S
					weight_down[i][w] = weight_down[i][w] / (1 - StepSize_dif);
				}
				objchange_up[i][s] = objfunc(DataSetA, weight_up,i) - objfunc_old;
				objchange_down[i][s] = objfunc(DataSetA, weight_down,i) - objfunc_old;
				objchange[i][s] = (objchange_up[i][s] - objchange_down[i][s])/2/StepSize_dif;
				
			}
		}
		//-----------------------------------------------------
		//(4).Calculate the Max gradient
		float objchange_max = 0;
		unsigned int objchange_maxID;
		//Looping
		for(int i = 0; i < PositionNum; i++)//I
		{
			for(int j = 0; j < ChannelNum; j++)//S
			{
				if(objchange_max<=fabs(objchange[i][j]) && ((objchange_up[i][j] * objchange_down[i][j]<0) || (objchange_up[i][j]<0 && objchange_down[i][j]<0)))
					objchange_max = fabs(objchange[i][j]);
			}
		}
		
		//-----------------------------------------------------
		//(5).Update the pars_mu: weight
		float weight_new[PositionNum][ChannelNum];
		memcpy(weight_new, weight, sizeof(weight));
		if(objchange_max!=0)
		{
			for(int i = 0; i < PositionNum; i++)
			{
				for(int j = 0; j < ChannelNum; j++)
				{
					if((objchange_up[i][j] * objchange_down[i][j]<0) || (objchange_up[i][j]<0 && objchange_down[i][j]<0))
					{
						//Iteration, down gradient
						weight_new[i][j] = weight[i][j] - objchange[i][j]/objchange_max * StepSize_cal;// * (1 - IterID/IterNum * 0.9);
					}
				}
			}
			//Normalization
			for(int i = 0; i < PositionNum; i++)
			{
				float weight_sum = 0;
				for(int j = 0; j < ChannelNum; j++)
				{
					weight_sum = weight_sum + weight_new[i][j];
				}
				for(int w = 0; w < ChannelNum; w++)
				{
					weight_new[i][w] = weight_new[i][w]/weight_sum;
					//cout<<weight_new[i][w]<<" "<<weight_tmp[i][w]<<endl;
				}
			}
		}
		memcpy(weight, weight_new, sizeof(weight_new));//Updating
		int t2 = clock();
		IterID = IterID + 1;
		cout<<"Time Elapsed: "<<(t2-t1)/1.e3<<" s"<<endl; 
	}
	fwrite(weight, sizeof(float), sizeof(weight)/sizeof(float), fp4);
	fclose(fp4);
	fclose(fp3);
	free(DataSetA);
	free(BMUID);
	free(BMUDistance);


int t_end = clock();
cout<<"All Time Elapsed: "<<(t_end-t_start)/1.e3<<" s"<<endl;
return 0;

}


//===========================================================================
//FUNCTION DEFINITION
//===========================================================================

//Griding Function，return minDistance and ID for events n
//算一个事例
//================================================================
BMU CalBMU(float signal[ChannelNum], float weight_value[PositionNum][ChannelNum], int BMUID_old)
{
	BMU a;//对象
	float EuclidDistance[PositionNum];//sum_S{(a_is - w_is)**2}[I]
	int BMUIDx_left, BMUIDx_right, BMUIDy_left, BMUIDy_right,BMUIDz_left, BMUIDz_right;
	//Looping the whole crastal
	if(BMUID_old==-1)
	{
		BMUIDx_left = 0; 
		BMUIDx_right = PositionNum_x - 1;
		BMUIDy_left = 0; 
		BMUIDy_right = PositionNum_y - 1;
		BMUIDz_left = 0; 
		BMUIDz_right = PositionNum_z - 1;
	}
	//Decrease the looping area, Just find the nearby grids
	else
	{
		int BMUIDz = BMUID_old/(PositionNum_x * PositionNum_y);
		int BMUIDy = (BMUID_old - BMUIDz * PositionNum_x * PositionNum_y)/PositionNum_x;
		int BMUIDx = BMUID_old - BMUIDz * PositionNum_x * PositionNum_y - BMUIDy * PositionNum_x;

		BMUIDx_left = BMUIDx-2;
		if(BMUIDx_left<0)
			BMUIDx_left = 0;

		BMUIDx_right = BMUIDx+2;
		if(BMUIDx_right>=PositionNum_x)
			BMUIDx_right = PositionNum_x - 1;

		BMUIDy_left = BMUIDy-2;
		if(BMUIDy_left<0)
			BMUIDy_left = 0;

		BMUIDy_right = BMUIDy+2;
		if(BMUIDy_right>=PositionNum_y)
			BMUIDy_right = PositionNum_y - 1;

		BMUIDz_left = BMUIDz-1;
		if(BMUIDz_left<0)
			BMUIDz_left = 0;

		BMUIDz_right = BMUIDz+1;
		if(BMUIDz_right>=PositionNum_z)
			BMUIDz_right = PositionNum_z - 1;

	}
	
	a.MinDistance = 10000;//初始值
	a.MinDistanceID = 0;
	//Griding the event:min_I{sum_S{(a_is - w_is)**2}} & MinDistanceID
	for(int yy = BMUIDy_left; yy <= BMUIDy_right; yy++)                                                  
	{
		for(int xx = BMUIDx_left; xx <= BMUIDx_right; xx++)
		{
			for(int zz = BMUIDz_left; zz <= BMUIDz_right; zz++)
			{
				int posID = zz * PositionNum_x * PositionNum_y + yy * PositionNum_x + xx;//位置编号：X->Y->Z
				EuclidDistance[posID] = 0;
				for(int s = 0; s < ChannelNum; s++)//计算位置posID处所有SIPM信号差(归一化光子数之差)的平方和
				{
					EuclidDistance[posID] = EuclidDistance[posID] + (signal[s] - weight_value[posID][s]) * (signal[s] - weight_value[posID][s]);
					//(a_is - w_is)**2
				}
				a.Distances[posID] = EuclidDistance[posID];
				if(EuclidDistance[posID]<=a.MinDistance)//寻找最小差别
				{
					a.MinDistance = EuclidDistance[posID];//min_I{sum_S{(a_is - w_is)**2}}
					a.MinDistanceID = posID;//对应的位置
				}
			}
		}
	}
	return a;
}


BMU CalBMU_Gradient(float signal[ChannelNum], float weight_value[PositionNum][ChannelNum], float distances[PositionNum], int GradientID)
{
	BMU a;//对象
	a.MinDistance = 10000;//初始值
	a.MinDistanceID = 0;
	float distances_tem[PositionNum] = {0};
	for(int s = 0; s < ChannelNum; s++)//计算位置posID处所有SIPM信号差(归一化光子数之差)的平方和
			{
				distances_tem[GradientID] = distances_tem[GradientID] + (signal[s] - weight_value[GradientID][s]) * (signal[s] - weight_value[GradientID][s]);
				//(a_is - w_is)**2
			}
	for(int i=0;i<PositionNum;i++){
		//cout<<i<<" "<<GradientID<<" "<<distances[GradientID]<<" "<<distances[i]<<endl;
		if(i != GradientID)
		{
			distances_tem[i] = distances[i];
		}
		
		if(distances_tem[i]<=a.MinDistance)//寻找最小差别
			{
				a.MinDistance = distances_tem[i];//min_I{sum_S{(a_is - w_is)**2}}
				a.MinDistanceID = i;//对应的位置
			}
	}
	return a;
	

}


//Cost Functions
//循环所有事例
//================================================================
float objfunc(float** DataSet, float weight_value[PositionNum][ChannelNum],int GradientID)
{
	//-----------------------------------------------------
	//Counting Gamma events in Grid[I = a.MinDistanceID]
	float Distri_cal[PositionNum] = {0};//Inverted IPD P[I]
	float MinDistance_sum[PositionNum] = {0};// sum_N[I]{min_I{sum_S{(a_is - w_is)**2}}}
	float Events_Clustered[PositionNum][ChannelNum] = {0};//Sum signal at [i][s]: sum_N[I](a_is) 

	const int NUM_segs = 24; //24 threads
	int EventNum_tem[NUM_segs] = {0};
	int EventNum_step = floor(EventNum / NUM_segs);
	for(int num = 0;num < NUM_segs;num++){
		EventNum_tem[num] = 0 + num * EventNum_step;
		//cout<<EventNum_tem[num]<<endl;
	}

	//thread 0
	float Distri_cal_tem[PositionNum] = {0};
	float MinDistance_sum_tem[PositionNum] = {0};
	float Events_Clustered_tem[PositionNum][ChannelNum] = {0};
	future<void> ft = async(std::launch::async, [&]{
			for(int n = EventNum_tem[0]; n < EventNum_tem[0] + EventNum_step; n++)//Looping events N
			{
				BMU a;
					
				if(GradientID == -1)
				{
					a = CalBMU(DataSet[n], weight_value, BMUID[n]);//Griding
				}
				else
				{
					a = CalBMU_Gradient(DataSet[n],weight_value,BMUDistance[n],GradientID); 
				}
				
				Distri_cal_tem[a.MinDistanceID] = Distri_cal_tem[a.MinDistanceID] + 1.0/EventNum;//Inverted IPD
				MinDistance_sum_tem[a.MinDistanceID] = MinDistance_sum_tem[a.MinDistanceID] + a.MinDistance;
				for(int s = 0; s<ChannelNum; s++)//Count events at [n][s]
				{
					Events_Clustered_tem[a.MinDistanceID][s] = Events_Clustered_tem[a.MinDistanceID][s] + DataSet[n][s];//sum(a_is) at grid [i][s]
				}
			}
	});

	//thread 1
	float Distri_cal_tem1[PositionNum] = {0};
	float MinDistance_sum_tem1[PositionNum] = {0};
	float Events_Clustered_tem1[PositionNum][ChannelNum] = {0};
	future<void> ft1 = async(std::launch::async, [&]{
			for(int n = EventNum_tem[1]; n < EventNum_tem[1] + EventNum_step; n++)//Looping events N
		{
			BMU a1;
				
			if(GradientID == -1)
			{
				a1 = CalBMU(DataSet[n], weight_value, BMUID[n]);//Griding
			}
			else
			{
				a1 = CalBMU_Gradient(DataSet[n],weight_value,BMUDistance[n],GradientID);
			}
			
			Distri_cal_tem1[a1.MinDistanceID] = Distri_cal_tem1[a1.MinDistanceID] + 1.0/EventNum;//Inverted IPD
			MinDistance_sum_tem1[a1.MinDistanceID] = MinDistance_sum_tem1[a1.MinDistanceID] + a1.MinDistance;
			for(int s = 0; s<ChannelNum; s++)//Count events at [n][s]
			{
				Events_Clustered_tem1[a1.MinDistanceID][s] = Events_Clustered_tem1[a1.MinDistanceID][s] + DataSet[n][s];//sum(a_is) at grid [i][s]
			}
		}
	});

	//thread 2
	float Distri_cal_tem2[PositionNum] = {0};
	float MinDistance_sum_tem2[PositionNum] = {0};
	float Events_Clustered_tem2[PositionNum][ChannelNum] = {0};
	future<void> ft2 = async(std::launch::async, [&]{
			for(int n = EventNum_tem[2]; n < EventNum_tem[2] + EventNum_step; n++)//Looping events N
		{
			BMU a2;
				
			if(GradientID == -1)
			{
				a2 = CalBMU(DataSet[n], weight_value, BMUID[n]);//Griding
			}
			else
			{
				a2 = CalBMU_Gradient(DataSet[n],weight_value,BMUDistance[n],GradientID);
			}
			
			Distri_cal_tem2[a2.MinDistanceID] = Distri_cal_tem2[a2.MinDistanceID] + 1.0/EventNum;//Inverted IPD
			MinDistance_sum_tem2[a2.MinDistanceID] = MinDistance_sum_tem2[a2.MinDistanceID] + a2.MinDistance;
			for(int s = 0; s<ChannelNum; s++)//Count events at [n][s]
			{
				Events_Clustered_tem2[a2.MinDistanceID][s] = Events_Clustered_tem2[a2.MinDistanceID][s] + DataSet[n][s];//sum(a_is) at grid [i][s]
			}
		}
	});

	//thread 3
	float Distri_cal_tem3[PositionNum] = {0};
	float MinDistance_sum_tem3[PositionNum] = {0};
	float Events_Clustered_tem3[PositionNum][ChannelNum] = {0};
	future<void> ft3 = async(std::launch::async, [&]{
			for(int n = EventNum_tem[3]; n < EventNum_tem[3] + EventNum_step; n++)//Looping events N
		{
			BMU a3;
				
			if(GradientID == -1)
			{
				a3 = CalBMU(DataSet[n], weight_value, BMUID[n]);//Griding
			}
			else
			{
				a3 = CalBMU_Gradient(DataSet[n],weight_value,BMUDistance[n],GradientID);
			}
			
			Distri_cal_tem3[a3.MinDistanceID] = Distri_cal_tem3[a3.MinDistanceID] + 1.0/EventNum;//Inverted IPD
			MinDistance_sum_tem3[a3.MinDistanceID] = MinDistance_sum_tem3[a3.MinDistanceID] + a3.MinDistance;
			for(int s = 0; s<ChannelNum; s++)//Count events at [n][s]
			{
				Events_Clustered_tem3[a3.MinDistanceID][s] = Events_Clustered_tem3[a3.MinDistanceID][s] + DataSet[n][s];//sum(a_is) at grid [i][s]
			}
		}
	});

	//thread 4
	float Distri_cal_tem4[PositionNum] = {0};
	float MinDistance_sum_tem4[PositionNum] = {0};
	float Events_Clustered_tem4[PositionNum][ChannelNum] = {0};
	future<void> ft4 = async(std::launch::async, [&]{
			for(int n = EventNum_tem[4]; n < EventNum_tem[4] + EventNum_step; n++)//Looping events N
			{
				BMU a4;
					
				if(GradientID == -1)
				{
					a4 = CalBMU(DataSet[n], weight_value, BMUID[n]);//Griding
				}
				else
				{
					a4 = CalBMU_Gradient(DataSet[n],weight_value,BMUDistance[n],GradientID); 
				}
				
				Distri_cal_tem4[a4.MinDistanceID] = Distri_cal_tem4[a4.MinDistanceID] + 1.0/EventNum;//Inverted IPD
				MinDistance_sum_tem4[a4.MinDistanceID] = MinDistance_sum_tem4[a4.MinDistanceID] + a4.MinDistance;
				for(int s = 0; s<ChannelNum; s++)//Count events at [n][s]
				{
					Events_Clustered_tem4[a4.MinDistanceID][s] = Events_Clustered_tem4[a4.MinDistanceID][s] + DataSet[n][s];//sum(a_is) at grid [i][s]
				}
			}
	});

	//thread 5
	float Distri_cal_tem5[PositionNum] = {0};
	float MinDistance_sum_tem5[PositionNum] = {0};
	float Events_Clustered_tem5[PositionNum][ChannelNum] = {0};
	future<void> ft5 = async(std::launch::async, [&]{
			for(int n = EventNum_tem[5]; n < EventNum_tem[5] + EventNum_step; n++)//Looping events N
		{
			BMU a5;
				
			if(GradientID == -1)
			{
				a5 = CalBMU(DataSet[n], weight_value, BMUID[n]);//Griding
			}
			else
			{
				a5 = CalBMU_Gradient(DataSet[n],weight_value,BMUDistance[n],GradientID);
			}
			
			Distri_cal_tem5[a5.MinDistanceID] = Distri_cal_tem5[a5.MinDistanceID] + 1.0/EventNum;//Inverted IPD
			MinDistance_sum_tem5[a5.MinDistanceID] = MinDistance_sum_tem5[a5.MinDistanceID] + a5.MinDistance;
			for(int s = 0; s<ChannelNum; s++)//Count events at [n][s]
			{
				Events_Clustered_tem5[a5.MinDistanceID][s] = Events_Clustered_tem5[a5.MinDistanceID][s] + DataSet[n][s];//sum(a_is) at grid [i][s]
			}
		}
	});

	//thread 6
	float Distri_cal_tem6[PositionNum] = {0};
	float MinDistance_sum_tem6[PositionNum] = {0};
	float Events_Clustered_tem6[PositionNum][ChannelNum] = {0};
	future<void> ft6 = async(std::launch::async, [&]{
			for(int n = EventNum_tem[6]; n < EventNum_tem[6] + EventNum_step; n++)//Looping events N
		{
			BMU a6;
				
			if(GradientID == -1)
			{
				a6 = CalBMU(DataSet[n], weight_value, BMUID[n]);//Griding
			}
			else
			{
				a6 = CalBMU_Gradient(DataSet[n],weight_value,BMUDistance[n],GradientID);
			}
			
			Distri_cal_tem6[a6.MinDistanceID] = Distri_cal_tem6[a6.MinDistanceID] + 1.0/EventNum;//Inverted IPD
			MinDistance_sum_tem6[a6.MinDistanceID] = MinDistance_sum_tem6[a6.MinDistanceID] + a6.MinDistance;
			for(int s = 0; s<ChannelNum; s++)//Count events at [n][s]
			{
				Events_Clustered_tem6[a6.MinDistanceID][s] = Events_Clustered_tem6[a6.MinDistanceID][s] + DataSet[n][s];//sum(a_is) at grid [i][s]
			}
		}
	});

	//thread 7
	float Distri_cal_tem7[PositionNum] = {0};
	float MinDistance_sum_tem7[PositionNum] = {0};
	float Events_Clustered_tem7[PositionNum][ChannelNum] = {0};
	future<void> ft7 = async(std::launch::async, [&]{
			for(int n = EventNum_tem[7]; n < EventNum_tem[7] + EventNum_step; n++)//Looping events N
		{
			BMU a7;
				
			if(GradientID == -1)
			{
				a7 = CalBMU(DataSet[n], weight_value, BMUID[n]);//Griding
			}
			else
			{
				a7 = CalBMU_Gradient(DataSet[n],weight_value,BMUDistance[n],GradientID);
			}
			
			Distri_cal_tem7[a7.MinDistanceID] = Distri_cal_tem7[a7.MinDistanceID] + 1.0/EventNum;//Inverted IPD
			MinDistance_sum_tem7[a7.MinDistanceID] = MinDistance_sum_tem7[a7.MinDistanceID] + a7.MinDistance;
			for(int s = 0; s<ChannelNum; s++)//Count events at [n][s]
			{
				Events_Clustered_tem7[a7.MinDistanceID][s] = Events_Clustered_tem7[a7.MinDistanceID][s] + DataSet[n][s];//sum(a_is) at grid [i][s]
			}
		}
	});

	//thread 8
	float Distri_cal_tem8[PositionNum] = {0};
	float MinDistance_sum_tem8[PositionNum] = {0};
	float Events_Clustered_tem8[PositionNum][ChannelNum] = {0};
	future<void> ft8 = async(std::launch::async, [&]{
			for(int n = EventNum_tem[8]; n < EventNum_tem[8] + EventNum_step; n++)//Looping events N
			{
				BMU a8;
					
				if(GradientID == -1)
				{
					a8 = CalBMU(DataSet[n], weight_value, BMUID[n]);//Griding
				}
				else
				{
					a8 = CalBMU_Gradient(DataSet[n],weight_value,BMUDistance[n],GradientID); 
				}
				
				Distri_cal_tem8[a8.MinDistanceID] = Distri_cal_tem8[a8.MinDistanceID] + 1.0/EventNum;//Inverted IPD
				MinDistance_sum_tem8[a8.MinDistanceID] = MinDistance_sum_tem8[a8.MinDistanceID] + a8.MinDistance;
				for(int s = 0; s<ChannelNum; s++)//Count events at [n][s]
				{
					Events_Clustered_tem8[a8.MinDistanceID][s] = Events_Clustered_tem8[a8.MinDistanceID][s] + DataSet[n][s];//sum(a_is) at grid [i][s]
				}
			}
	});

	//thread 9
	float Distri_cal_tem9[PositionNum] = {0};
	float MinDistance_sum_tem9[PositionNum] = {0};
	float Events_Clustered_tem9[PositionNum][ChannelNum] = {0};
	future<void> ft9 = async(std::launch::async, [&]{
			for(int n = EventNum_tem[9]; n < EventNum_tem[9] + EventNum_step; n++)//Looping events N
		{
			BMU a9;
				
			if(GradientID == -1)
			{
				a9 = CalBMU(DataSet[n], weight_value, BMUID[n]);//Griding
			}
			else
			{
				a9 = CalBMU_Gradient(DataSet[n],weight_value,BMUDistance[n],GradientID);
			}
			
			Distri_cal_tem9[a9.MinDistanceID] = Distri_cal_tem9[a9.MinDistanceID] + 1.0/EventNum;//Inverted IPD
			MinDistance_sum_tem9[a9.MinDistanceID] = MinDistance_sum_tem9[a9.MinDistanceID] + a9.MinDistance;
			for(int s = 0; s<ChannelNum; s++)//Count events at [n][s]
			{
				Events_Clustered_tem9[a9.MinDistanceID][s] = Events_Clustered_tem9[a9.MinDistanceID][s] + DataSet[n][s];//sum(a_is) at grid [i][s]
			}
		}
	});

	//thread 10
	float Distri_cal_tem10[PositionNum] = {0};
	float MinDistance_sum_tem10[PositionNum] = {0};
	float Events_Clustered_tem10[PositionNum][ChannelNum] = {0};
	future<void> ft10 = async(std::launch::async, [&]{
			for(int n = EventNum_tem[10]; n < EventNum_tem[10] + EventNum_step; n++)//Looping events N
		{
			BMU a10;
				
			if(GradientID == -1)
			{
				a10 = CalBMU(DataSet[n], weight_value, BMUID[n]);//Griding
			}
			else
			{
				a10 = CalBMU_Gradient(DataSet[n],weight_value,BMUDistance[n],GradientID);
			}
			
			Distri_cal_tem10[a10.MinDistanceID] = Distri_cal_tem10[a10.MinDistanceID] + 1.0/EventNum;//Inverted IPD
			MinDistance_sum_tem10[a10.MinDistanceID] = MinDistance_sum_tem10[a10.MinDistanceID] + a10.MinDistance;
			for(int s = 0; s<ChannelNum; s++)//Count events at [n][s]
			{
				Events_Clustered_tem10[a10.MinDistanceID][s] = Events_Clustered_tem10[a10.MinDistanceID][s] + DataSet[n][s];//sum(a_is) at grid [i][s]
			}
		}
	});

	//thread 11
	float Distri_cal_tem11[PositionNum] = {0};
	float MinDistance_sum_tem11[PositionNum] = {0};
	float Events_Clustered_tem11[PositionNum][ChannelNum] = {0};
	future<void> ft11 = async(std::launch::async, [&]{
			for(int n = EventNum_tem[11]; n < EventNum_tem[11] + EventNum_step; n++)//Looping events N
		{
			BMU a11;
				
			if(GradientID == -1)
			{
				a11 = CalBMU(DataSet[n], weight_value, BMUID[n]);//Griding
			}
			else
			{
				a11 = CalBMU_Gradient(DataSet[n],weight_value,BMUDistance[n],GradientID);
			}
			
			Distri_cal_tem11[a11.MinDistanceID] = Distri_cal_tem11[a11.MinDistanceID] + 1.0/EventNum;//Inverted IPD
			MinDistance_sum_tem11[a11.MinDistanceID] = MinDistance_sum_tem11[a11.MinDistanceID] + a11.MinDistance;
			for(int s = 0; s<ChannelNum; s++)//Count events at [n][s]
			{
				Events_Clustered_tem11[a11.MinDistanceID][s] = Events_Clustered_tem11[a11.MinDistanceID][s] + DataSet[n][s];//sum(a_is) at grid [i][s]
			}
		}
	});

	//thread 12
	float Distri_cal_tem12[PositionNum] = {0};
	float MinDistance_sum_tem12[PositionNum] = {0};
	float Events_Clustered_tem12[PositionNum][ChannelNum] = {0};
	future<void> ft12 = async(std::launch::async, [&]{
			for(int n = EventNum_tem[12]; n < EventNum_tem[12] + EventNum_step; n++)//Looping events N
			{
				BMU a12;
					
				if(GradientID == -1)
				{
					a12 = CalBMU(DataSet[n], weight_value, BMUID[n]);//Griding
				}
				else
				{
					a12 = CalBMU_Gradient(DataSet[n],weight_value,BMUDistance[n],GradientID); 
				}
				
				Distri_cal_tem12[a12.MinDistanceID] = Distri_cal_tem12[a12.MinDistanceID] + 1.0/EventNum;//Inverted IPD
				MinDistance_sum_tem12[a12.MinDistanceID] = MinDistance_sum_tem12[a12.MinDistanceID] + a12.MinDistance;
				for(int s = 0; s<ChannelNum; s++)//Count events at [n][s]
				{
					Events_Clustered_tem12[a12.MinDistanceID][s] = Events_Clustered_tem12[a12.MinDistanceID][s] + DataSet[n][s];//sum(a_is) at grid [i][s]
				}
			}
	});

	//thread 13
	float Distri_cal_tem13[PositionNum] = {0};
	float MinDistance_sum_tem13[PositionNum] = {0};
	float Events_Clustered_tem13[PositionNum][ChannelNum] = {0};
	future<void> ft13 = async(std::launch::async, [&]{
			for(int n = EventNum_tem[13]; n < EventNum_tem[13] + EventNum_step; n++)//Looping events N
		{
			BMU a13;
				
			if(GradientID == -1)
			{
				a13 = CalBMU(DataSet[n], weight_value, BMUID[n]);//Griding
			}
			else
			{
				a13 = CalBMU_Gradient(DataSet[n],weight_value,BMUDistance[n],GradientID);
			}
			
			Distri_cal_tem13[a13.MinDistanceID] = Distri_cal_tem13[a13.MinDistanceID] + 1.0/EventNum;//Inverted IPD
			MinDistance_sum_tem13[a13.MinDistanceID] = MinDistance_sum_tem13[a13.MinDistanceID] + a13.MinDistance;
			for(int s = 0; s<ChannelNum; s++)//Count events at [n][s]
			{
				Events_Clustered_tem13[a13.MinDistanceID][s] = Events_Clustered_tem13[a13.MinDistanceID][s] + DataSet[n][s];//sum(a_is) at grid [i][s]
			}
		}
	});

	//thread 14
	float Distri_cal_tem14[PositionNum] = {0};
	float MinDistance_sum_tem14[PositionNum] = {0};
	float Events_Clustered_tem14[PositionNum][ChannelNum] = {0};
	future<void> ft14 = async(std::launch::async, [&]{
			for(int n = EventNum_tem[14]; n < EventNum_tem[14] + EventNum_step; n++)//Looping events N
		{
			BMU a14;
				
			if(GradientID == -1)
			{
				a14 = CalBMU(DataSet[n], weight_value, BMUID[n]);//Griding
			}
			else
			{
				a14 = CalBMU_Gradient(DataSet[n],weight_value,BMUDistance[n],GradientID);
			}
			
			Distri_cal_tem14[a14.MinDistanceID] = Distri_cal_tem14[a14.MinDistanceID] + 1.0/EventNum;//Inverted IPD
			MinDistance_sum_tem14[a14.MinDistanceID] = MinDistance_sum_tem14[a14.MinDistanceID] + a14.MinDistance;
			for(int s = 0; s<ChannelNum; s++)//Count events at [n][s]
			{
				Events_Clustered_tem14[a14.MinDistanceID][s] = Events_Clustered_tem14[a14.MinDistanceID][s] + DataSet[n][s];//sum(a_is) at grid [i][s]
			}
		}
	});

	//thread 15
	float Distri_cal_tem15[PositionNum] = {0};
	float MinDistance_sum_tem15[PositionNum] = {0};
	float Events_Clustered_tem15[PositionNum][ChannelNum] = {0};
	future<void> ft15 = async(std::launch::async, [&]{
			for(int n = EventNum_tem[15]; n < EventNum_tem[15] + EventNum_step; n++)//Looping events N
		{
			BMU a15;
				
			if(GradientID == -1)
			{
				a15 = CalBMU(DataSet[n], weight_value, BMUID[n]);//Griding
			}
			else
			{
				a15 = CalBMU_Gradient(DataSet[n],weight_value,BMUDistance[n],GradientID);
			}
			
			Distri_cal_tem15[a15.MinDistanceID] = Distri_cal_tem15[a15.MinDistanceID] + 1.0/EventNum;//Inverted IPD
			MinDistance_sum_tem15[a15.MinDistanceID] = MinDistance_sum_tem15[a15.MinDistanceID] + a15.MinDistance;
			for(int s = 0; s<ChannelNum; s++)//Count events at [n][s]
			{
				Events_Clustered_tem15[a15.MinDistanceID][s] = Events_Clustered_tem15[a15.MinDistanceID][s] + DataSet[n][s];//sum(a_is) at grid [i][s]
			}
		}
	});

	//thread 16
	float Distri_cal_tem16[PositionNum] = {0};
	float MinDistance_sum_tem16[PositionNum] = {0};
	float Events_Clustered_tem16[PositionNum][ChannelNum] = {0};
	future<void> ft16 = async(std::launch::async, [&]{
			for(int n = EventNum_tem[16]; n < EventNum_tem[16] + EventNum_step; n++)//Looping events N
			{
				BMU a16;
					
				if(GradientID == -1)
				{
					a16 = CalBMU(DataSet[n], weight_value, BMUID[n]);//Griding
				}
				else
				{
					a16 = CalBMU_Gradient(DataSet[n],weight_value,BMUDistance[n],GradientID); 
				}
				
				Distri_cal_tem16[a16.MinDistanceID] = Distri_cal_tem16[a16.MinDistanceID] + 1.0/EventNum;//Inverted IPD
				MinDistance_sum_tem16[a16.MinDistanceID] = MinDistance_sum_tem16[a16.MinDistanceID] + a16.MinDistance;
				for(int s = 0; s<ChannelNum; s++)//Count events at [n][s]
				{
					Events_Clustered_tem16[a16.MinDistanceID][s] = Events_Clustered_tem16[a16.MinDistanceID][s] + DataSet[n][s];//sum(a_is) at grid [i][s]
				}
			}
	});

	//thread 17
	float Distri_cal_tem17[PositionNum] = {0};
	float MinDistance_sum_tem17[PositionNum] = {0};
	float Events_Clustered_tem17[PositionNum][ChannelNum] = {0};
	future<void> ft17 = async(std::launch::async, [&]{
			for(int n = EventNum_tem[17]; n < EventNum_tem[17] + EventNum_step; n++)//Looping events N
		{
			BMU a17;
				
			if(GradientID == -1)
			{
				a17 = CalBMU(DataSet[n], weight_value, BMUID[n]);//Griding
			}
			else
			{
				a17 = CalBMU_Gradient(DataSet[n],weight_value,BMUDistance[n],GradientID);
			}
			
			Distri_cal_tem17[a17.MinDistanceID] = Distri_cal_tem17[a17.MinDistanceID] + 1.0/EventNum;//Inverted IPD
			MinDistance_sum_tem17[a17.MinDistanceID] = MinDistance_sum_tem17[a17.MinDistanceID] + a17.MinDistance;
			for(int s = 0; s<ChannelNum; s++)//Count events at [n][s]
			{
				Events_Clustered_tem17[a17.MinDistanceID][s] = Events_Clustered_tem17[a17.MinDistanceID][s] + DataSet[n][s];//sum(a_is) at grid [i][s]
			}
		}
	});

	//thread 18
	float Distri_cal_tem18[PositionNum] = {0};
	float MinDistance_sum_tem18[PositionNum] = {0};
	float Events_Clustered_tem18[PositionNum][ChannelNum] = {0};
	future<void> ft18 = async(std::launch::async, [&]{
			for(int n = EventNum_tem[18]; n < EventNum_tem[18] + EventNum_step; n++)//Looping events N
		{
			BMU a18;
				
			if(GradientID == -1)
			{
				a18 = CalBMU(DataSet[n], weight_value, BMUID[n]);//Griding
			}
			else
			{
				a18 = CalBMU_Gradient(DataSet[n],weight_value,BMUDistance[n],GradientID);
			}
			
			Distri_cal_tem18[a18.MinDistanceID] = Distri_cal_tem18[a18.MinDistanceID] + 1.0/EventNum;//Inverted IPD
			MinDistance_sum_tem18[a18.MinDistanceID] = MinDistance_sum_tem18[a18.MinDistanceID] + a18.MinDistance;
			for(int s = 0; s<ChannelNum; s++)//Count events at [n][s]
			{
				Events_Clustered_tem18[a18.MinDistanceID][s] = Events_Clustered_tem18[a18.MinDistanceID][s] + DataSet[n][s];//sum(a_is) at grid [i][s]
			}
		}
	});

	//thread 19
	float Distri_cal_tem19[PositionNum] = {0};
	float MinDistance_sum_tem19[PositionNum] = {0};
	float Events_Clustered_tem19[PositionNum][ChannelNum] = {0};
	future<void> ft19 = async(std::launch::async, [&]{
			for(int n = EventNum_tem[19]; n < EventNum_tem[19] + EventNum_step; n++)//Looping events N
		{
			BMU a19;
				
			if(GradientID == -1)
			{
				a19 = CalBMU(DataSet[n], weight_value, BMUID[n]);//Griding
			}
			else
			{
				a19 = CalBMU_Gradient(DataSet[n],weight_value,BMUDistance[n],GradientID);
			}
			
			Distri_cal_tem19[a19.MinDistanceID] = Distri_cal_tem19[a19.MinDistanceID] + 1.0/EventNum;//Inverted IPD
			MinDistance_sum_tem19[a19.MinDistanceID] = MinDistance_sum_tem19[a19.MinDistanceID] + a19.MinDistance;
			for(int s = 0; s<ChannelNum; s++)//Count events at [n][s]
			{
				Events_Clustered_tem19[a19.MinDistanceID][s] = Events_Clustered_tem19[a19.MinDistanceID][s] + DataSet[n][s];//sum(a_is) at grid [i][s]
			}
		}
	});

	//thread 20
	float Distri_cal_tem20[PositionNum] = {0};
	float MinDistance_sum_tem20[PositionNum] = {0};
	float Events_Clustered_tem20[PositionNum][ChannelNum] = {0};
	future<void> ft20 = async(std::launch::async, [&]{
			for(int n = EventNum_tem[20]; n < EventNum_tem[20] + EventNum_step; n++)//Looping events N
			{
				BMU a20;
					
				if(GradientID == -1)
				{
					a20 = CalBMU(DataSet[n], weight_value, BMUID[n]);//Griding
				}
				else
				{
					a20 = CalBMU_Gradient(DataSet[n],weight_value,BMUDistance[n],GradientID); 
				}
				
				Distri_cal_tem20[a20.MinDistanceID] = Distri_cal_tem20[a20.MinDistanceID] + 1.0/EventNum;//Inverted IPD
				MinDistance_sum_tem20[a20.MinDistanceID] = MinDistance_sum_tem20[a20.MinDistanceID] + a20.MinDistance;
				for(int s = 0; s<ChannelNum; s++)//Count events at [n][s]
				{
					Events_Clustered_tem20[a20.MinDistanceID][s] = Events_Clustered_tem20[a20.MinDistanceID][s] + DataSet[n][s];//sum(a_is) at grid [i][s]
				}
			}
	});

	//thread 21
	float Distri_cal_tem21[PositionNum] = {0};
	float MinDistance_sum_tem21[PositionNum] = {0};
	float Events_Clustered_tem21[PositionNum][ChannelNum] = {0};
	future<void> ft21 = async(std::launch::async, [&]{
			for(int n = EventNum_tem[21]; n < EventNum_tem[21] + EventNum_step; n++)//Looping events N
		{
			BMU a21;
				
			if(GradientID == -1)
			{
				a21 = CalBMU(DataSet[n], weight_value, BMUID[n]);//Griding
			}
			else
			{
				a21 = CalBMU_Gradient(DataSet[n],weight_value,BMUDistance[n],GradientID);
			}
			
			Distri_cal_tem21[a21.MinDistanceID] = Distri_cal_tem21[a21.MinDistanceID] + 1.0/EventNum;//Inverted IPD
			MinDistance_sum_tem21[a21.MinDistanceID] = MinDistance_sum_tem21[a21.MinDistanceID] + a21.MinDistance;
			for(int s = 0; s<ChannelNum; s++)//Count events at [n][s]
			{
				Events_Clustered_tem21[a21.MinDistanceID][s] = Events_Clustered_tem21[a21.MinDistanceID][s] + DataSet[n][s];//sum(a_is) at grid [i][s]
			}
		}
	});

	//thread 22
	float Distri_cal_tem22[PositionNum] = {0};
	float MinDistance_sum_tem22[PositionNum] = {0};
	float Events_Clustered_tem22[PositionNum][ChannelNum] = {0};
	future<void> ft22 = async(std::launch::async, [&]{
			for(int n = EventNum_tem[22]; n < EventNum_tem[22] + EventNum_step; n++)//Looping events N
		{
			BMU a22;
				
			if(GradientID == -1)
			{
				a22 = CalBMU(DataSet[n], weight_value, BMUID[n]);//Griding
			}
			else
			{
				a22 = CalBMU_Gradient(DataSet[n],weight_value,BMUDistance[n],GradientID);
			}
			
			Distri_cal_tem22[a22.MinDistanceID] = Distri_cal_tem22[a22.MinDistanceID] + 1.0/EventNum;//Inverted IPD
			MinDistance_sum_tem22[a22.MinDistanceID] = MinDistance_sum_tem22[a22.MinDistanceID] + a22.MinDistance;
			for(int s = 0; s<ChannelNum; s++)//Count events at [n][s]
			{
				Events_Clustered_tem22[a22.MinDistanceID][s] = Events_Clustered_tem22[a22.MinDistanceID][s] + DataSet[n][s];//sum(a_is) at grid [i][s]
			}
		}
	});

	//thread 23
	float Distri_cal_tem23[PositionNum] = {0};
	float MinDistance_sum_tem23[PositionNum] = {0};
	float Events_Clustered_tem23[PositionNum][ChannelNum] = {0};
	future<void> ft23 = async(std::launch::async, [&]{
			for(int n = EventNum_tem[23]; n < EventNum; n++)//  !!!!!!!!!!!!!!!!!!!!!!!!
		{
			BMU a23;
				
			if(GradientID == -1)
			{
				a23 = CalBMU(DataSet[n], weight_value, BMUID[n]);//Griding
			}
			else
			{
				a23 = CalBMU_Gradient(DataSet[n],weight_value,BMUDistance[n],GradientID);
			}
			
			Distri_cal_tem23[a23.MinDistanceID] = Distri_cal_tem23[a23.MinDistanceID] + 1.0/EventNum;//Inverted IPD
			MinDistance_sum_tem23[a23.MinDistanceID] = MinDistance_sum_tem23[a23.MinDistanceID] + a23.MinDistance;
			for(int s = 0; s<ChannelNum; s++)//Count events at [n][s]
			{
				Events_Clustered_tem23[a23.MinDistanceID][s] = Events_Clustered_tem23[a23.MinDistanceID][s] + DataSet[n][s];//sum(a_is) at grid [i][s]
			}
		}
	});

	ft.wait();
    ft1.wait();
	ft2.wait();
    ft3.wait();
	ft4.wait();
    ft5.wait();
	ft6.wait();
    ft7.wait();
	ft8.wait();
    ft9.wait();
	ft10.wait();
    ft11.wait();
	ft12.wait();
    ft13.wait();
	ft14.wait();
    ft15.wait();
	ft16.wait();
    ft17.wait();
	ft18.wait();
    ft19.wait();
	ft20.wait();
    ft21.wait();
	ft22.wait();
    ft23.wait();

	for(int i = 0;i<PositionNum;i++){
		Distri_cal[i] += Distri_cal_tem[i] +Distri_cal_tem1[i] +Distri_cal_tem2[i] +Distri_cal_tem3[i]
					  +  Distri_cal_tem4[i] +Distri_cal_tem5[i] +Distri_cal_tem6[i] +Distri_cal_tem7[i]
					  +  Distri_cal_tem8[i] +Distri_cal_tem9[i] +Distri_cal_tem10[i] +Distri_cal_tem11[i]
					  +  Distri_cal_tem12[i] +Distri_cal_tem13[i] +Distri_cal_tem14[i] +Distri_cal_tem15[i]
					  +  Distri_cal_tem16[i] +Distri_cal_tem17[i] +Distri_cal_tem18[i] +Distri_cal_tem19[i]
					  +  Distri_cal_tem20[i] +Distri_cal_tem21[i] +Distri_cal_tem22[i] +Distri_cal_tem23[i];
		MinDistance_sum[i] += MinDistance_sum_tem[i] + MinDistance_sum_tem1[i] + MinDistance_sum_tem2[i] + MinDistance_sum_tem3[i]
		                   +  MinDistance_sum_tem4[i] + MinDistance_sum_tem5[i] + MinDistance_sum_tem6[i] + MinDistance_sum_tem7[i]
						   +  MinDistance_sum_tem8[i] + MinDistance_sum_tem9[i] + MinDistance_sum_tem10[i] + MinDistance_sum_tem11[i]
		                   +  MinDistance_sum_tem12[i] + MinDistance_sum_tem13[i] + MinDistance_sum_tem14[i] + MinDistance_sum_tem15[i]
						   +  MinDistance_sum_tem16[i] + MinDistance_sum_tem17[i] + MinDistance_sum_tem18[i] + MinDistance_sum_tem19[i]
		                   +  MinDistance_sum_tem20[i] + MinDistance_sum_tem21[i] + MinDistance_sum_tem22[i] + MinDistance_sum_tem23[i];
		for(int s = 0;s<ChannelNum;s++){
			Events_Clustered[i][s] += Events_Clustered_tem[i][s] +Events_Clustered_tem1[i][s] +Events_Clustered_tem2[i][s] +Events_Clustered_tem3[i][s]
			                       +  Events_Clustered_tem4[i][s] +Events_Clustered_tem5[i][s] +Events_Clustered_tem6[i][s] +Events_Clustered_tem7[i][s]
								   +  Events_Clustered_tem8[i][s] +Events_Clustered_tem9[i][s] +Events_Clustered_tem10[i][s] +Events_Clustered_tem11[i][s]
			                       +  Events_Clustered_tem12[i][s] +Events_Clustered_tem13[i][s] +Events_Clustered_tem14[i][s] +Events_Clustered_tem15[i][s]
								   +  Events_Clustered_tem16[i][s] +Events_Clustered_tem17[i][s] +Events_Clustered_tem18[i][s] +Events_Clustered_tem19[i][s]
			                       +  Events_Clustered_tem20[i][s] +Events_Clustered_tem21[i][s] +Events_Clustered_tem22[i][s] +Events_Clustered_tem23[i][s];
		}
	}





	//cout<<endl;
	//-----------------------------------------------------
	//Calculating f3
	//f3 = sum_IS(sum(a_is)/N(is) - w_is)
	float objfunc_3 = 0;
	for(int i = 0; i<PositionNum; i++)//I
	{
		for(int s = 0; s<ChannelNum; s++)//S
		{
			if(Distri_cal[i]!=0)
			{
				Events_Clustered[i][s] = Events_Clustered[i][s]/(Distri_cal[i] * EventNum);//Average signal at [i][s]:sum(a_is) / N(is)
				objfunc_3 = objfunc_3 + (Events_Clustered[i][s] - weight_value[i][s]) * (Events_Clustered[i][s] - weight_value[i][s]);
			}
			else
				objfunc_3 = objfunc_3 + (1 - weight_value[i][s]) * (1 - weight_value[i][s]);
		}
	}
	//-----------------------------------------------------
	//Calculating f1,f2
	float objfunc_1 = 0;//f1 = sum_I{sum_N[I]{min_I{sum_S{(a_is - w_is)**2}}}}
	float objfunc_2 = 0;//f2 = sum_I{(^P_i-P_i)**2}
	for(int i = 0; i < PositionNum; i++)
	{
		objfunc_1 = objfunc_1 + MinDistance_sum[i];
		objfunc_2 = objfunc_2 + (Distri_cal[i] - Distri_real[i]) * (Distri_cal[i] - Distri_real[i]);
	}
	//-----------------------------------------------------
	//Add Weights to f1,f2,f3
	//cout<<objfunc_1/EventNum<<" "<<objfunc_2 * sqrt(EventNum * 1.0)<<" "<<objfunc_3 * sqrt(EventNum * 1.0)<<endl;
	//return objfunc_1/EventNum + (1 * objfunc_2 + objfunc_3) * sqrt(EventNum * 1.0);
	//return objfunc_1/EventNum + (1 * objfunc_2 ) * sqrt(EventNum * 1.0);//f1 + f2
	return objfunc_1/EventNum + (1 * objfunc_2 );//f1 + f2N
	//return (objfunc_1/EventNum) *  log(1 * objfunc_2 );//f1 + f2log
	//return (1 * objfunc_2 ) * sqrt(EventNum * 1.0);//f2
	
}

 
 #endif


