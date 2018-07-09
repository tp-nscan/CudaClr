#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <wtypes.h>
#include <curand.h>
#include <stdexcept>
#include <sstream>
#include <string>


extern "C" __declspec(dllexport) BSTR DllMakeCublasHandle(void **dev_handle);

extern "C" __declspec(dllexport) BSTR DllDestroyCublasHandle(void *dev_handle);


//C = α op ( A ) op ( B ) + β C
extern "C" __declspec(dllexport) BSTR DllcublasSgemm(
	void *dev_handle,   //handle to the cuBLAS library context.
	int transa,         //cublasOperation_t val for matrix A
	int transb,         //cublasOperation_t val for matrix B
	int m,              //number of rows of matrix op(A) and C.
	int n,              //number of columns of matrix op(B) and C.
	int k,              //number of columns of op(A) and rows of op(B).
	const float *alpha, //host or device pointer for α
	const float *A,     // array of dimensions lda x k with lda>=max(1,m) if transa == CUBLAS_OP_N and lda x m with lda>=max(1,k) otherwise.
	int lda,            // leading dimension of two-dimensional array used to store the matrix A.
	const float *B,     // array of dimension ldb x n with ldb>=max(1,k) if transb == CUBLAS_OP_N and ldb x k with ldb>=max(1,n) otherwise.
	int ldb,            // leading dimension of two-dimensional array used to store matrix B.
	const float *beta,  // host or device pointer
	float *C,           // array of dimensions ldc x n with ldc>=max(1,m).        
	int ldc);           // leading dimension of a two-dimensional array used to store the matrix C.