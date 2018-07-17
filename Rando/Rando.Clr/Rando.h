#pragma once
#include "stdafx.h"

extern "C" __declspec(dllimport) BSTR DllMakeGenerator64(int seed);

extern "C" __declspec(dllimport) BSTR DllMakeGenerator32(int seed);

extern "C" __declspec(dllimport) BSTR DllDestroyGenerator();

extern "C" __declspec(dllimport) BSTR DllMakeRandomInts(int *devPtr, unsigned int numRands);

extern "C" __declspec(dllimport) BSTR DllMakeUniformRands(float *devPtr, unsigned int numRands);

extern "C" __declspec(dllimport) BSTR DllMakeNormalRands(float *devPtr, unsigned int numRands, float mean, float stdev);

extern "C" __declspec(dllimport) BSTR DllMakeLogNormalRands(float *devPtr, unsigned int numRands, float mean, float stdev);

//The curandGeneratePoisson() function is used to generate Poisson - distributed integer 
//values based on a Poisson distribution with the given lambda.
extern "C" __declspec(dllimport) BSTR DllMakePoissonRands(unsigned int *devPtr, unsigned int numRands, double lambda);
