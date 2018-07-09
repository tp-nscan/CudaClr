#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <wtypes.h>
#include <stdexcept>
#include <sstream>
#include <string>
#include <cublas_v2.h>
#include "..\..\Common\ClrUtils.h"

extern "C" __declspec(dllexport) BSTR DllMakeCublasHandle(void **dev_handle)
{
	std::string funcName = "DllMakeCublasHandle";
	try
	{
		cublasHandle_t **gp = (cublasHandle_t **)dev_handle;
		cublasStatus_t cublasStatus = cublasCreate(*gp);

		if (cublasStatus != cudaSuccess) {
			return CublasStatusBSTR(cublasStatus, funcName);
		}
		return BSTR();
	}
	catch (std::runtime_error &e)
	{
		std::string err = e.what();
		return RuntimeErrBSTR(err, funcName);
	}
}


extern "C" __declspec(dllexport) BSTR DllDestroyCublasHandle(void *dev_handle)
{
	std::string funcName = "DllMakeCublasHandle";
	try
	{
		cublasHandle_t *gp = (cublasHandle_t *)dev_handle;
		cublasStatus_t cublasStatus = cublasDestroy(*gp);

		if (cublasStatus != cudaSuccess) {
			return CublasStatusBSTR(cublasStatus, funcName);
		}
		return BSTR();
	}
	catch (std::runtime_error &e)
	{
		std::string err = e.what();
		return RuntimeErrBSTR(err, funcName);
	}
}

extern "C" __declspec(dllexport) BSTR DllcublasSgemm(
	void *dev_handle,
	int transa,
	int transb,
	int m,
	int n,
	int k,
	const float *alpha,
	const float *A,
	int lda,
	const float *B,
	int ldb,
	const float *beta,
	float *C,
	int ldc)
{
	std::string funcName = "DllcublasSgemm";
	try
	{
		const float aa = 1.0f;
		const float bb = 0.0f;
		cublasHandle_t *gp = (cublasHandle_t *)dev_handle;
		cublasStatus_t cublasStatus = cublasSgemm(
			*gp,
			(cublasOperation_t)transa,
			(cublasOperation_t)transb,
			m, n, k, &aa, A, lda, B, ldb, &bb, C, ldc);

		if (cublasStatus != cudaSuccess) {
			return CublasStatusBSTR(cublasStatus, funcName);
		}
		return BSTR();
	}
	catch (std::runtime_error &e)
	{
		std::string err = e.what();
		return RuntimeErrBSTR(err, funcName);
	}
}