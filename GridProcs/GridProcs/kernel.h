#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void copyKernel(int *out, const int *in);

__global__ void GolKernel(int *out, const int *in, const int span);