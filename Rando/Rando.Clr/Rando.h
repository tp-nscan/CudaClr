#pragma once
#include "stdafx.h"


extern "C" __declspec(dllimport) BSTR DllMakeGenerator(void **genPtr, int seed);

extern "C" __declspec(dllimport) BSTR DllDestroyGenerator(void *genPtr);

extern "C" __declspec(dllimport) BSTR
	DllMakeUniformRands(float *devPtr, void *genPtr, unsigned int numRands);

extern "C" __declspec(dllimport) BSTR
	DllMakeNormalRands(float *devPtr, void *genPtr, unsigned int numRands, float mean, float stdev);

extern "C" __declspec(dllimport) BSTR
	DllMakeLogNormalRands(float *devPtr, void *genPtr, unsigned int numRands, float mean, float stdev);

//The curandGeneratePoisson() function is used to generate Poisson - distributed integer 
//values based on a Poisson distribution with the given lambda.
extern "C" __declspec(dllimport) BSTR
	DllMakePoissonRands(int *devPtr, void *genPtr, unsigned int numRands, double lambda);


