#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define gSpan 16  // linear system size
const int gArea = gSpan * gSpan;


__global__ void copyKernel(int *out, int *in);

__global__ void GolKernel(int *out, int *in, unsigned int span);

__global__ void Ca9fKernel(float *output, float *input, float *rands, unsigned int span, float step_size, float noise);

__global__ void AltKernel(int *data, unsigned int span, int alt, int value);

__global__ void AltKernelCopy(int *dataOut, int *dataIn, unsigned int span, int alt, int value);

__global__ void AltIsingKernel(int *data, float *rands, float temp, unsigned int span, int alt);

__global__ void AltIsingKernelCopy(int *dataOut, int *dataIn, float *rands, float temp, unsigned int span, int alt);

__global__ void device_function_init_YK(double d_t, int* d_spin, int* d_bond, double* d_random_data, unsigned int* d_label);

__global__ void IsingKernel(int *data, float *rands, unsigned int span, int alt, float t1, float t2, float t3, float t4);
