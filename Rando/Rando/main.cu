#include <stdio.h>
#include "Rando.h"


////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
//int main(int argc, char **argv)
//{
//	int seed = 123;
//	int arrayLen = 10;
//	printf("Hi");
//	curandGenerator_t *g1 = (curandGenerator_t *)malloc(sizeof(curandGenerator_t));
//	BSTR res = DllMakeGenerator(&g1, seed);
//
//	curandGenerator_t *g2 = (curandGenerator_t *)malloc(sizeof(curandGenerator_t));
//	res = DllMakeGenerator(&g2, seed);
//
//	float *devFloats1 = (float *)malloc(sizeof(float));
//	float *devFloats2 = (float *)malloc(sizeof(float));
//
//	res = DllMallocFloatsOnDevice(&devFloats1, arrayLen);
//	res = DllMallocFloatsOnDevice(&devFloats2, arrayLen);
//
//
//	res = DllMakeUniformRands(devFloats1, g1, arrayLen);
//	res = DllMakeUniformRands(devFloats2, g2, arrayLen);
//
//	float *hostFloats1 = (float *)malloc(sizeof(float) * arrayLen);
//	float *hostFloats2 = (float *)malloc(sizeof(float) * arrayLen);
//
//	res = DllCopyFloatsFromDevice(hostFloats1, devFloats1, arrayLen);
//	res = DllCopyFloatsFromDevice(hostFloats2, devFloats2, arrayLen);
//
//	res = DllMakeUniformRands(devFloats1, g1, arrayLen);
//	res = DllMakeUniformRands(devFloats2, g2, arrayLen);
//
//
//	res = DllCopyFloatsFromDevice(hostFloats1, devFloats1, arrayLen);
//	res = DllCopyFloatsFromDevice(hostFloats2, devFloats2, arrayLen);
//
//}
