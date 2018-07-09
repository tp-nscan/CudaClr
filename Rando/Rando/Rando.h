#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <wtypes.h>
#include <curand.h>
#include <stdexcept>
#include <sstream>
#include <string>


extern "C" __declspec(dllexport) BSTR DllMakeGenerator(void **genPtr, int seed);

extern "C" __declspec(dllexport) BSTR DllDestroyGenerator(void *genPtr);

extern "C" __declspec(dllexport) BSTR 
	DllMakeUniformRands(float *devPtr, void *genPtr, unsigned int numRands);

extern "C" __declspec(dllexport) BSTR 
	DllMakeNormalRands(float *devPtr, void *genPtr, unsigned int numRands, float mean, float stdev);

extern "C" __declspec(dllexport) BSTR
	DllMakeLogNormalRands(float *devPtr, void *genPtr, unsigned int numRands, float mean, float stdev);

//The curandGeneratePoisson() function is used to generate Poisson - distributed integer 
//values based on a Poisson distribution with the given lambda.
extern "C" __declspec(dllexport) BSTR
	DllMakePoissonRands(unsigned int *devPtr, void *genPtr, unsigned int numRands, double lambda);
