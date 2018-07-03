#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <wtypes.h>
#include <stdexcept>
#include "CuArray.h"
#include <sstream>
#include <string>
#include "..\..\Common\ClrUtils.h"

////////////////////////////////////////
// DLL interface
////////////////////////////////////////


BSTR DllTestRuntimeErr()
{
	std::string funcName = "DllTestRuntimeErr";
	char *runtimeErr = "Runtime err msg.";
	std::string err = runtimeErr;
	return RuntimeErrBSTR(err, funcName);
}

BSTR DllTestCudaStatusErr()
{
	std::string funcName = "DllTestCudaStatusErr";
	cudaError_t cudaStatus = cudaErrorMissingConfiguration;

	return CudaStatusBSTR(cudaStatus, funcName);
}

BSTR DllMallocOnDevice(void **dev_ints, unsigned int bytes)
{
	std::string funcName = "DllMallocOnDevice";
	try
	{
		cudaError_t cudaStatus = cudaMalloc(dev_ints, bytes);
		if (cudaStatus != cudaSuccess) {
			return CudaStatusBSTR(cudaStatus, funcName);
		}
		return BSTR();
	}
	catch (std::runtime_error &e)
	{
		std::string err = e.what();
		return RuntimeErrBSTR(err, funcName);
	}
}


BSTR DllReleaseDevicePtr(void *dev_ptr)
{
	std::string funcName = "DllReleaseDevicePtr";

	try
	{
		cudaError_t cudaStatus = cudaFree(dev_ptr);
		cudaFree(dev_ptr);

		if (cudaStatus != cudaSuccess) {
			return CudaStatusBSTR(cudaStatus, funcName);
		}
		return BSTR();
	}
	catch (std::runtime_error &e)
	{
		std::string err = e.what();
		return RuntimeErrBSTR(err, funcName);
	}
}


BSTR DllCopyToDevice(void *dev_ints, const void *host_ints, unsigned int bytes)
{
	std::string funcName = "DllCopyToDevice";
	try
	{
		cudaError_t cudaStatus = cudaMemcpy(dev_ints, host_ints, bytes, cudaMemcpyHostToDevice);

		if (cudaStatus != cudaSuccess) {
			return CudaStatusBSTR(cudaStatus, funcName);
		}
		return BSTR();
	}
	catch (std::runtime_error &e)
	{
		std::string err = e.what();
		return RuntimeErrBSTR(err, funcName);
	}
}


BSTR DllCopyFromDevice(void *host_ints, const void *dev_ints, unsigned int bytes)
{
	std::string funcName = "DllCopyFromDevice";
	try
	{
		cudaError_t cudaStatus = cudaMemcpy(host_ints, dev_ints, bytes, cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			return CudaStatusBSTR(cudaStatus, funcName);
		}
		return BSTR();
	}
	catch (std::runtime_error &e)
	{
		std::string err = e.what();
		return RuntimeErrBSTR(err, funcName);
	}
}


BSTR DllCopyDeviceToDevice(void *dest_ints, const void *src_ints, unsigned int bytes)
{
	std::string funcName = "DllCopyDeviceToDevice";
	try
	{
		cudaError_t cudaStatus = cudaMemcpy(dest_ints, src_ints, bytes, cudaMemcpyDeviceToDevice);
		if (cudaStatus != cudaSuccess) {
			return CudaStatusBSTR(cudaStatus, funcName);
		}
		return BSTR();
	}
	catch (std::runtime_error &e)
	{
		std::string err = e.what();
		return RuntimeErrBSTR(err, funcName);
	}
}


BSTR DllDeviceSynchronize()
{
	std::string funcName = "DllDeviceSynchronize";
	try
	{
		cudaError_t cudaStatus = cudaDeviceSynchronize(); 
		if (cudaStatus != cudaSuccess) {
			return CudaStatusBSTR(cudaStatus, funcName);
		}
		return BSTR();
	}
	catch (std::runtime_error &e)
	{
		std::string err = e.what();
		return RuntimeErrBSTR(err, funcName);
	}
}


BSTR DllResetDevice()
{
	std::string funcName = "DllResetDevice";
	try
	{
		cudaError_t cudaStatus = cudaDeviceReset();
		if (cudaStatus != cudaSuccess) {
			return CudaStatusBSTR(cudaStatus, funcName);
		}
		return BSTR();
	}
	catch (std::runtime_error &e)
	{
		std::string err = e.what();
		return RuntimeErrBSTR(err, funcName);
	}
}