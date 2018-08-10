#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void LinearReduceKernel(int *d_out, const int *d_in, unsigned int length);

__global__ void BlockReduce_32_Kernel(int *d_out, const int *d_in);

__global__ void BlockReduce_16_Kernel(int *d_out, const int *d_in);
