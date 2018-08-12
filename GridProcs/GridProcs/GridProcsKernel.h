#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void GolKernel(int *out, int *in, unsigned int span);

__global__ void AltKernel(int *data, unsigned int span, int alt, int value);

__global__ void AltKernelCopy(int *dataOut, int *dataIn, unsigned int span, int alt, int value);

__global__ void MetroIsingKernel(int *dataOut, int *dataIn, float *rands, float temp, unsigned int span, int alt);

__global__ void IsingKernel(int *dataOut, int *dataIn, float *rands, unsigned int span, int alt, float t2, float t4);

__global__ void IsingKernelEnergy(int *energyOut, int *dataIn, unsigned int span);

__global__ void IsingKernelPlusEnergy(int *dataOut, int *energyOut, int *dataIn, float *rands, unsigned int span, int alt, float *thresh);

__global__ void k_RandBlockPick(int *dataOut, unsigned int *rands, unsigned int block_bits);

__global__ void k_IsingRb(int *dataOut, int *energyOut, unsigned int *index_rands, float *temp_rands, unsigned int block_size, float t2, float t4);

__global__ void device_function_init_YK(double d_t, int* d_spin, int* d_bond, double* d_random_data, unsigned int* d_label);

__global__ void k_Thermo(float *dataOut, float *dataIn, unsigned int span, int alt, float rate, int fixed_colA, int fixed_colB);