#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <wtypes.h>
#include <curand.h>
#include <stdexcept>
#include <sstream>
#include <string>
#include "Utils.h"


////////////////////////////////////////
// DLL interface
////////////////////////////////////////

extern "C" __declspec(dllexport) BSTR DllMakeGenerator(void **gen, int seed)
{
	std::string funcName = "DllMakeGenerator";
	try
	{
		curandGenerator_t **gp = (curandGenerator_t **)gen;
		curandStatus_t curandStatus =
			curandCreateGenerator(*gp, CURAND_RNG_PSEUDO_XORWOW);

		if (curandStatus != CURAND_STATUS_SUCCESS) {
			return CurandStatusBSTR(curandStatus, funcName);
		}
		return BSTR();
	}
	catch (std::runtime_error &e)
	{
		std::string err = e.what();
		return RuntimeErrBSTR(err, funcName);
	}
}


extern "C" __declspec(dllexport) BSTR DllDestroyGenerator(void *gen)
{
	std::string funcName = "DllDestroyGenerator";
	try
	{
		curandGenerator_t *gp = (curandGenerator_t *)gen;
		curandStatus_t curandStatus = curandDestroyGenerator(*gp);

		if (curandStatus != CURAND_STATUS_SUCCESS) {
			return CurandStatusBSTR(curandStatus, funcName);
		}
		return BSTR();
	}
	catch (std::runtime_error &e)
	{
		std::string err = e.what();
		return RuntimeErrBSTR(err, funcName);
	}
}


extern "C" __declspec(dllexport) BSTR
DllMakeUniformRands(float *devPtr, void *gen, unsigned int numRands)
{
	std::string funcName = "DllMakeUniformRands";
	try
	{
		curandGenerator_t *gp = (curandGenerator_t *)gen;
		curandStatus_t curandStatus =
			curandGenerateUniform(*gp, devPtr, numRands);

		if (curandStatus != CURAND_STATUS_SUCCESS) {
			return CurandStatusBSTR(curandStatus, funcName);
		}
		return BSTR();
	}
	catch (std::runtime_error &e)
	{
		std::string err = e.what();
		return RuntimeErrBSTR(err, funcName);
	}
}

extern "C" __declspec(dllexport) BSTR
DllMakeNormalRands(float *devPtr, void *gen, unsigned int numRands, float mean, float stdev)
{
	std::string funcName = "DllMakeNormalRands";
	try
	{
		curandGenerator_t *gp = (curandGenerator_t *)gen;
		curandStatus_t curandStatus =
			curandGenerateNormal(*gp, devPtr, numRands, mean, stdev);

		if (curandStatus != CURAND_STATUS_SUCCESS) {
			return CurandStatusBSTR(curandStatus, funcName);
		}
		return BSTR();
	}
	catch (std::runtime_error &e)
	{
		std::string err = e.what();
		return RuntimeErrBSTR(err, funcName);
	}
}

extern "C" __declspec(dllexport) BSTR
DllMakeLogNormalRands(float *devPtr, void *gen, unsigned int numRands, float mean, float stdev)
{
	std::string funcName = "DllMakeLogNormalRands";
	try
	{
		curandGenerator_t *gp = (curandGenerator_t *)gen;
		curandStatus_t curandStatus =
			curandGenerateLogNormal(*gp, devPtr, numRands, mean, stdev);

		if (curandStatus != CURAND_STATUS_SUCCESS) {
			return CurandStatusBSTR(curandStatus, funcName);
		}
		return BSTR();
	}
	catch (std::runtime_error &e)
	{
		std::string err = e.what();
		return RuntimeErrBSTR(err, funcName);
	}
}

extern "C" __declspec(dllexport) BSTR
DllMakePoissonRands(unsigned int *devPtr, void *gen, unsigned int numRands, double lambda)
{
	std::string funcName = "DllMakePoissonRands";
	try
	{
		curandGenerator_t *gp = (curandGenerator_t *)gen;
		curandStatus_t curandStatus =
			curandGeneratePoisson(*gp, devPtr, numRands, lambda);

		if (curandStatus != CURAND_STATUS_SUCCESS) {
			return CurandStatusBSTR(curandStatus, funcName);
		}
		return BSTR();
	}
	catch (std::runtime_error &e)
	{
		std::string err = e.what();
		return RuntimeErrBSTR(err, funcName);
	}
}