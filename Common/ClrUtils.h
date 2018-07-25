#pragma once
#include <wtypes.h>
#include <string>
#include <curand.h>
#include "cuda_runtime.h"
#include "Cublas.h"

BSTR StdStrToBSTR(std::string what);

BSTR RuntimeErrBSTR(std::string err_msg, std::string func_name);

BSTR CudaStatusBSTR(cudaError_t cuda_status, std::string func_name);

const char* CudaStatusToChars(cudaError_t status);

BSTR CurandStatusBSTR(curandStatus_t cuda_status, std::string func_name);

const char* CurandStatusToChars(curandStatus_t status);

BSTR CublasStatusBSTR(cublasStatus_t cuda_status, std::string func_name);

const char* CublasStatusToChars(cublasStatus_t status);

unsigned int SqrtPow2Lb(unsigned int rhs);