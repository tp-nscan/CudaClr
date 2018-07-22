#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <wtypes.h>
#include <curand.h>
#include <stdexcept>
#include <sstream>
#include <string>

extern "C" __declspec(dllexport) BSTR DllMakeGenerator64(int seed);

extern "C" __declspec(dllexport) BSTR DllMakeGenerator32(int seed);

extern "C" __declspec(dllexport) BSTR DllDestroyGenerator();

extern "C" __declspec(dllexport) BSTR DllMakeRandomInts(int *devPtr, unsigned int numRands);

extern "C" __declspec(dllexport) BSTR DllMakeUniformRands(float *devPtr, unsigned int numRands);

extern "C" __declspec(dllexport) BSTR DllMakeNormalRands(float *devPtr, unsigned int numRands, float mean, float stdev);

extern "C" __declspec(dllexport) BSTR DllMakeLogNormalRands(float *devPtr, unsigned int numRands, float mean, float stdev);

extern "C" __declspec(dllexport) BSTR DllMakePoissonRands(unsigned int *devPtr, unsigned int numRands, double lambda);

extern "C" __declspec(dllexport) BSTR DllMakeUniformDoubleRands(double *devPtr, unsigned int numRands);
