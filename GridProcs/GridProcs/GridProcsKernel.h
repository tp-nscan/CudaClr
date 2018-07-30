#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void GolKernel(int *out, int *in, unsigned int span);

__global__ void AltKernel(int *data, unsigned int span, int alt, int value);

__global__ void AltKernelCopy(int *dataOut, int *dataIn, unsigned int span, int alt, int value);

__global__ void MetroIsingKernel(int *data, float *rands, float temp, unsigned int span, int alt);

__global__ void MetroIsingKernelCopy(int *dataOut, int *dataIn, float *rands, float temp, unsigned int span, int alt);

__global__ void IsingKernel(int *dataOut, int *dataIn, float *rands, unsigned int span, int alt, float t2, float t4);

__global__ void IsingKernelPlusEnergy(int *dataOut, int *energyOut, int *dataIn, float *rands, unsigned int span, int alt, float t2, float t4);

__global__ void IsingKernelVariableTemp(int *dataOut, int *energyOut, int *dataIn, float *rands, unsigned int span, int alt, float t2, float t4, bool flip);

__global__ void LinearReduceKernel(int *d_out, const int *d_in, unsigned int length);

__global__ void BlockReduce_16_Kernel(int *d_out, const int *d_in);

__global__ void BlockReduce_32_Kernel(int *d_out, const int *d_in);

__global__ void device_function_init_YK(double d_t, int* d_spin, int* d_bond, double* d_random_data, unsigned int* d_label);

