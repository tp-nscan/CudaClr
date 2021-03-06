#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <wtypes.h>
#include <stdexcept>
#include "CuArray.h"
#include <sstream>
#include <string>
#include "..\..\Common\ClrUtils.h"
#include "CuArrayKernels.h"

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


BSTR DllRunLinearAddIntsKernel(int *d_out, const int *d_in, unsigned int length_in, unsigned int length_out)
{
	std::string funcName = "DllRunLinearAddIntsKernel";
	try
	{
		LinearAddIntsKernel<<<length_out, 512>>>(d_out, d_in, length_in);
		return BSTR();
	}
	catch (std::runtime_error &e)
	{
		std::string err = e.what();
		return RuntimeErrBSTR(err, funcName);
	}
}


BSTR DllRunBlockAddInts_32_Kernel(int *d_out, const int *d_in, unsigned int span)
{
	std::string funcName = "DllRunBlockAddInts_32_Kernel";
	try
	{
		dim3 b = dim3(span / 32, span / 32);
		dim3 t = dim3(32, 32);
		BlockAddInts_32_Kernel<<<b, t>>>(d_out, d_in);

		return BSTR();
	}
	catch (std::runtime_error &e)
	{
		std::string err = e.what();
		return RuntimeErrBSTR(err, funcName);
	}
}


BSTR DllRunBlockAddInts_16_Kernel(int *d_out, const int *d_in, unsigned int span)
{
	std::string funcName = "DllRunBlockAddInts_16_Kernel";
	try
	{
		dim3 b = dim3(span / 16, span / 16);
		dim3 t = dim3(16, 16);
		BlockAddInts_16_Kernel << <b, t >> >(d_out, d_in);
		return BSTR();
	}
	catch (std::runtime_error &e)
	{
		std::string err = e.what();
		return RuntimeErrBSTR(err, funcName);
	}
}

BSTR DllRunLinearAddFloatsKernel(float *d_out, const float *d_in, unsigned int length_in, unsigned int length_out)
{
	std::string funcName = "DllRunLinearAddFloatsKernel";
	try
	{
		LinearAddFloatsKernel <<<length_out, 512>>>(d_out, d_in, length_in);
		return BSTR();
	}
	catch (std::runtime_error &e)
	{
		std::string err = e.what();
		return RuntimeErrBSTR(err, funcName);
	}
}


BSTR DllRunBlockAddFloats_32_Kernel(float *d_out, const float *d_in, unsigned int span)
{
	std::string funcName = "DllRunBlockAddFloats_32_Kernel";
	try
	{
		dim3 b = dim3(span / 32, span / 32);
		dim3 t = dim3(32, 32);
		BlockAddFloats_32_Kernel <<<b, t>>>(d_out, d_in);

		return BSTR();
	}
	catch (std::runtime_error &e)
	{
		std::string err = e.what();
		return RuntimeErrBSTR(err, funcName);
	}
}


BSTR DllRunBlockAddFloats_16_Kernel(float *d_out, const float *d_in, unsigned int span)
{
	std::string funcName = "DllRunBlockAddFloats_16_Kernel";
	try
	{
		dim3 b = dim3(span / 16, span / 16);
		dim3 t = dim3(16, 16);
		BlockAddFloats_16_Kernel <<<b, t>>>(d_out, d_in);
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