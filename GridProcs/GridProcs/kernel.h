#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void copyKernel(int *out, const int *in);

__global__ void GolKernel(int *out, const int *in, const int span);

__global__ void Ca9fKernel(float *output, const float *input, float *rands, const int span, float step_size, float noise);