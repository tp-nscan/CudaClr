#pragma once
#include "stdafx.h"

extern "C" __declspec(dllimport) BSTR DllRunGolKernel(int *dev_out, int *dev_in, unsigned int span);

extern "C" __declspec(dllimport) BSTR DllRunAltKernel(int *data, unsigned int span, int alt, int value);

extern "C" __declspec(dllimport) BSTR DllRunAltKernelCopy(int *dataOut, int *dataIn, unsigned int span, int alt, int value);

extern "C" __declspec(dllimport) BSTR DllRunMetroIsingKernel(int *dataOut, int *dataIn, float *rands, float temp, unsigned int span, int alt);

extern "C" __declspec(dllimport) BSTR DllRunIsingKernel(int *dataOut, int *dataIn, float *rands, unsigned int span, int alt, float t2, float t4);

extern "C" __declspec(dllimport) BSTR DllRunIsingKernelEnergy(int *energyOut, int *dataIn, unsigned int span);

extern "C" __declspec(dllimport) BSTR DllRunIsingKernelPlusEnergy(int *dataOut, int *energyOut, int *dataIn, float *rands, unsigned int span, int alt, float *thresh);

extern "C" __declspec(dllimport) BSTR DllRun_k_RandBlockPick(int *dataOut, unsigned int *rands, unsigned int block_size, unsigned int blocks_per_span);

extern "C" __declspec(dllimport) BSTR DllRun_k_IsingRb(int *dataOut, int *energyOut, unsigned int *index_rands, float *temp_rands, unsigned int block_size, unsigned int blocks_per_span, float t2, float t4);

extern "C" __declspec(dllimport) BSTR DllRundevice_function_init_YK(double d_t, int* d_spin, int* d_bond, double* d_random_data, unsigned int* d_label);

extern "C" __declspec(dllimport) BSTR DllRun_k_Thermo(float *dataOut, float *dataIn, unsigned int span, int alt, float rate);