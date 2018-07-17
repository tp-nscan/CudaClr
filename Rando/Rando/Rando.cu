#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <wtypes.h>
#include <curand.h>
#include <stdexcept>
#include <sstream>
#include <string>
#include "..\..\Common\ClrUtils.h"


////////////////////////////////////////
// DLL interface
////////////////////////////////////////

curandGenerator_t _curandGenerator;


extern "C" __declspec(dllexport) BSTR DllMakeGenerator64(int seed)
{
	std::string funcName = "DllMakeGenerator64";
	try
	{
		curandStatus_t curandStatus = curandCreateGenerator(&_curandGenerator, CURAND_RNG_PSEUDO_XORWOW);
		if (curandStatus != CURAND_STATUS_SUCCESS) {
			std::string ctx = funcName + ".curandCreateGenerator";
			return CurandStatusBSTR(curandStatus, ctx);
		}

		cudaError_t cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			std::string ctx = funcName + ".cudaDeviceSynchronize_1";
			return CudaStatusBSTR(cudaStatus, ctx);
		}

		curandStatus = curandSetPseudoRandomGeneratorSeed(_curandGenerator, (long long)seed);
		if (cudaStatus != cudaSuccess) {
			std::string ctx = funcName + ".curandSetPseudoRandomGeneratorSeed";
			return CudaStatusBSTR(cudaStatus, ctx);
		}

		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			std::string ctx = funcName + ".cudaDeviceSynchronize_2";
			return CudaStatusBSTR(cudaStatus, ctx);
		}

		return BSTR();
	}
	catch (std::runtime_error &e)
	{
		std::string err = e.what();
		return RuntimeErrBSTR(err, funcName);
	}
}


extern "C" __declspec(dllexport) BSTR DllMakeGenerator32(int seed)
{
	std::string funcName = "DllMakeGenerator32";
	try
	{
		curandStatus_t curandStatus = curandCreateGenerator(&_curandGenerator, CURAND_RNG_PSEUDO_MRG32K3A);

		if (curandStatus != CURAND_STATUS_SUCCESS) {
			std::string ctx = funcName + ".curandCreateGenerator";
			return CurandStatusBSTR(curandStatus, ctx);
		}
		cudaError_t cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			std::string ctx = funcName + ".cudaDeviceSynchronize_1";
			return CudaStatusBSTR(cudaStatus, ctx);
		}

		curandStatus = curandSetPseudoRandomGeneratorSeed(_curandGenerator, (long long)seed);
		if (cudaStatus != cudaSuccess) {
			std::string ctx = funcName + ".curandSetPseudoRandomGeneratorSeed";
			return CudaStatusBSTR(cudaStatus, ctx);
		}

		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			std::string ctx = funcName + ".cudaDeviceSynchronize_2";
			return CudaStatusBSTR(cudaStatus, ctx);
		}

		return BSTR();
	}
	catch (std::runtime_error &e)
	{
		std::string err = e.what();
		return RuntimeErrBSTR(err, funcName);
	}
}


extern "C" __declspec(dllexport) BSTR DllDestroyGenerator()
{
	std::string funcName = "DllDestroyGenerator";
	try
	{
		curandStatus_t curandStatus = curandDestroyGenerator(_curandGenerator);

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


extern "C" __declspec(dllexport) BSTR DllMakeRandomInts(unsigned int *devPtr, unsigned int numRands)
{
	std::string funcName = "DllMakeRandomInts";
	try
	{
		curandStatus_t curandStatus = curandGenerateLongLong(_curandGenerator, (unsigned long long *)devPtr, numRands/4);

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
	DllMakeUniformRands(float *devPtr, unsigned int numRands)
{
	std::string funcName = "DllMakeUniformRands";
	try
	{
		curandStatus_t curandStatus = curandGenerateUniform(_curandGenerator, devPtr, numRands);

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


extern "C" __declspec(dllexport) BSTR DllMakeNormalRands(float *devPtr, unsigned int numRands, float mean, float stdev)
{
	std::string funcName = "DllMakeNormalRands";
	try
	{
		curandStatus_t curandStatus = curandGenerateNormal(_curandGenerator, devPtr, numRands, mean, stdev);

		if (curandStatus != CURAND_STATUS_SUCCESS) {
			return CurandStatusBSTR(curandStatus, funcName);
		}
		cudaError_t cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			std::string ctx = funcName + ".cudaDeviceSynchronize_1";
			return CudaStatusBSTR(cudaStatus, ctx);
		}

		return BSTR();
	}
	catch (std::runtime_error &e)
	{
		std::string err = e.what();
		return RuntimeErrBSTR(err, funcName);
	}
}


extern "C" __declspec(dllexport) BSTR DllMakeLogNormalRands(float *devPtr, unsigned int numRands, float mean, float stdev)
{
	std::string funcName = "DllMakeLogNormalRands";
	try
	{
		curandStatus_t curandStatus =
			curandGenerateLogNormal(_curandGenerator, devPtr, numRands, mean, stdev);

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

extern "C" __declspec(dllexport) BSTR DllMakePoissonRands(unsigned int *devPtr, unsigned int numRands, double lambda)
{
	std::string funcName = "DllMakePoissonRands";
	try
	{
		curandStatus_t curandStatus = curandGeneratePoisson(_curandGenerator, devPtr, numRands, lambda);

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