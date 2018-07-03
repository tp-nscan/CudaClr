#pragma once
#include <wtypes.h>
#include <string>
#include <curand.h>
#include "cuda_runtime.h"

BSTR StdStrToBSTR(std::string what);

BSTR CudaStatusBSTR(cudaError_t cuda_status, std::string func_name);

BSTR RuntimeErrBSTR(std::string err_msg, std::string func_name);

const char* CudaStatusToChars(cudaError_t status);

BSTR CurandStatusBSTR(curandStatus_t cuda_status, std::string func_name);

const char* CurandStatusToChars(curandStatus_t status);