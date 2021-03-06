#pragma once
#include "Rando.h"

using namespace System::Runtime::InteropServices;
using namespace System;

namespace RandoClr {
	public ref class RandoProcs
	{
	public:
		RandoProcs() {};

		String^ MakeGenerator64(int seed)
		{
			BSTR res = DllMakeGenerator64(seed);
			return gcnew String(res);
		}

		String^ MakeGenerator32(int seed)
		{
			BSTR res = DllMakeGenerator32(seed);
			return gcnew String(res);
		}

		String^ DestroyGenerator()
		{
			BSTR res = DllDestroyGenerator();
			return gcnew String(res);
		}

		String^ MakeRandomInts(IntPtr devPtr, unsigned int arraySize)
		{
			unsigned int *dPtr = (unsigned int *)devPtr.ToPointer();
			BSTR res = DllMakeRandomInts(dPtr, arraySize);
			return gcnew String(res);
		}

		String^ MakeUniformRands(IntPtr devPtr, unsigned int arraySize)
		{
			float *dPtr = (float *)devPtr.ToPointer();
			BSTR res = DllMakeUniformRands(dPtr, arraySize);
			return gcnew String(res);
		}

		String^ MakeNormalRands(IntPtr devPtr, unsigned int numRands, float mean, float stdev)
		{
			float *dPtr = (float *)devPtr.ToPointer();
			BSTR res = DllMakeNormalRands(dPtr, numRands, mean, stdev);
			return gcnew String(res);
		}

		String^ MakeLogNormalRands(IntPtr devPtr, unsigned int numRands, float mean, float stdev)
		{
			float *dPtr = (float *)devPtr.ToPointer();
			BSTR res = DllMakeLogNormalRands(dPtr, numRands, mean, stdev);
			return gcnew String(res);
		}

		String^ MakeLogNormalRands(IntPtr devPtr, unsigned int numRands, double lambda)
		{
			unsigned int *dPtr = (unsigned int *)devPtr.ToPointer();
			BSTR res = DllMakePoissonRands(dPtr, numRands, lambda);
			return gcnew String(res);
		}

		String^ MakeUniformDoubleRands(IntPtr devPtr, unsigned int numRands)
		{
			double *dPtr = (double *)devPtr.ToPointer();
			BSTR res = DllMakeUniformDoubleRands(dPtr, numRands);
			return gcnew String(res);
		}

	};
}
