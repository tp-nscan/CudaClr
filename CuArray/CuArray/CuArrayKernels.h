#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void LinearAddIntsKernel(int *d_out, const int *d_in, unsigned int length);

__global__ void BlockAddInts_32_Kernel(int *d_out, const int *d_in);

__global__ void BlockAddInts_16_Kernel(int *d_out, const int *d_in);

__global__ void LinearAddFloatsKernel(float *d_out, const float *d_in, unsigned int length);

__global__ void BlockAddFloats_32_Kernel(float *d_out, const float *d_in);

__global__ void BlockAddFloats_16_Kernel(float *d_out, const float *d_in);


