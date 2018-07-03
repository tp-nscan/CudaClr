#pragma once
#include "Rando.h"

using namespace System::Runtime::InteropServices;
using namespace System;

namespace RandoClr {
	public ref class Rando
	{
	public:
		Rando() {};

		String^ MakeGenerator(IntPtr %gen, int seed)
		{
			void *dPtr;
			BSTR res = DllMakeGenerator(&dPtr, seed);
			gen = IntPtr(dPtr);
			return gcnew String(res);
		}

		String^ DestroyGenerator(IntPtr gen)
		{
			void *gPtr = (void *)gen.ToPointer();
			BSTR res = DllDestroyGenerator(gPtr);
			return gcnew String(res);
		}

		String^ MakeUniformRands(IntPtr devPtr, IntPtr genPtr, unsigned int arraySize)
		{
			float *dPtr = (float *)devPtr.ToPointer();
			void *gPtr = (void *)genPtr.ToPointer();

			BSTR res = DllMakeUniformRands(dPtr, gPtr, arraySize);
			return gcnew String(res);
		}

		String^ MakeNormalRands(IntPtr devPtr, IntPtr genPtr, unsigned int numRands, float mean, float stdev)
		{
			float *dPtr = (float *)devPtr.ToPointer();
			void *gPtr = (void *)genPtr.ToPointer();

			BSTR res = DllMakeNormalRands(dPtr, gPtr, numRands, mean, stdev);

			return gcnew String(res);
		}

		String^ MakeLogNormalRands(IntPtr devPtr, IntPtr genPtr, unsigned int numRands, float mean, float stdev)
		{
			float *dPtr = (float *)devPtr.ToPointer();
			void *gPtr = (void *)genPtr.ToPointer();

			BSTR res = DllMakeLogNormalRands(dPtr, gPtr, numRands, mean, stdev);

			return gcnew String(res);
		}

		String^ MakeLogNormalRands(IntPtr devPtr, IntPtr genPtr, unsigned int numRands, double lambda)
		{
			int *dPtr = (int *)devPtr.ToPointer();
			void *gPtr = (void *)genPtr.ToPointer();

			BSTR res = DllMakePoissonRands(dPtr, gPtr, numRands, lambda);

			return gcnew String(res);
		}

	};
}
