#pragma once
#include "Cublas.h"

using namespace System;

namespace CublasClr {

	public enum class CublasOp {
		N = 0, //the non-transpose operation 
		T = 1, //the transpose operation
		C = 2  //the conjugate transpose operation
	};

	public ref class Cublas
	{
	public:
		Cublas() {};

		String^ MakeCublasHandle(IntPtr %cublas_handle)
		{
			void *dPtr;
			BSTR res = DllMakeCublasHandle(&dPtr);
			cublas_handle = IntPtr(dPtr);
			return gcnew String(res);
		}

		String^ DestroyCublasHandle(IntPtr cublas_handle)
		{
			void *_cublas_handle = (void *)cublas_handle.ToPointer();
			BSTR res = DllDestroyCublasHandle(_cublas_handle);
			return gcnew String(res);
		}

		String^ cublasSgemm(
					IntPtr cublas_handle,
					CublasOp transa,
					CublasOp transb,
					int m,
					int n,
					int k,
					const float alpha,
					IntPtr dev_A,
					int lda,
					IntPtr dev_B,
					int ldb,
					const float beta,
					IntPtr dev_C,
					int ldc)
		{
			const float host_alpha = alpha;
			const float host_beta = beta;
			int _transa = static_cast<int>(transa);
			int _transb = static_cast<int>(transb);

			void *_cublas_handle = (void *)cublas_handle.ToPointer();
			float *_dev_A = (float *)dev_A.ToPointer();
			float *_dev_B = (float *)dev_B.ToPointer();
			float *_dev_C = (float *)dev_C.ToPointer();

			BSTR res = DllcublasSgemm(
				_cublas_handle, _transa, _transb, m, n, k, 
				&host_alpha, _dev_A, lda, _dev_B, ldb, &host_beta, _dev_C, ldc);
	
			return gcnew String(res);
		}
	};
}
