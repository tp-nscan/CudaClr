#pragma once
#include "CuArray.h"

using namespace System::Runtime::InteropServices;
using namespace System;

namespace CuArrayClr {

	public ref class CudaArray
	{

	public:
		CudaArray() {};

		////////////////////////////////////////////
		/// Ints
		////////////////////////////////////////////

		String^ MallocIntsOnDevice(IntPtr %devPtr, unsigned int arraySize)
		{
			void *dPtr;
			BSTR res = DllMallocOnDevice(&dPtr, sizeof(int) * arraySize);
			devPtr = IntPtr(dPtr);
			String^ rv = gcnew String(res);
			SysFreeString(res);
			return rv;
		}

		String^ CopyIntsToDevice(cli::array<int>^ data, IntPtr devPtr, unsigned int arraySize)
		{
			int *hostPtr = (int *)malloc(sizeof(int) * arraySize);
			for (unsigned int i = 0; i < arraySize; i++)
			{
				hostPtr[i] = data[i];
			}

			BSTR res = DllCopyToDevice(devPtr.ToPointer(), hostPtr, sizeof(int) * arraySize);
			free(hostPtr);

			String^ rv = gcnew String(res);
			SysFreeString(res);
			return rv;
		}

		String^ CopyUIntsToDevice(cli::array<unsigned int>^ data, IntPtr devPtr, unsigned int arraySize)
		{
			unsigned int *hostPtr = (unsigned int *)malloc(sizeof(int) * arraySize);
			for (unsigned int i = 0; i < arraySize; i++)
			{
				hostPtr[i] = data[i];
			}

			BSTR res = DllCopyToDevice(devPtr.ToPointer(), hostPtr, sizeof(int) * arraySize);
			free(hostPtr);

			String^ rv = gcnew String(res);
			SysFreeString(res);
			return rv;
		}

		String^ CopyIntsFromDevice(cli::array<int>^ retlist, IntPtr devPtr, unsigned int arraySize)
		{
			int *hostPtr = (int *)malloc(sizeof(int) * arraySize);
			BSTR res = DllCopyFromDevice(hostPtr, devPtr.ToPointer(), sizeof(int) * arraySize);

			for (unsigned int i = 0; i < arraySize; i++)
			{
				retlist[i] = hostPtr[i];
			}
			free(hostPtr);

			String^ rv = gcnew String(res);
			SysFreeString(res);
			return rv;
		}

		String^ CopyUIntsFromDevice(cli::array<unsigned int>^ retlist, IntPtr devPtr, unsigned int arraySize)
		{
			unsigned int *hostPtr = (unsigned int *)malloc(sizeof(int) * arraySize);
			BSTR res = DllCopyFromDevice(hostPtr, devPtr.ToPointer(), sizeof(int) * arraySize);

			for (unsigned int i = 0; i < arraySize; i++)
			{
				retlist[i] = hostPtr[i];
			}
			free(hostPtr);

			String^ rv = gcnew String(res);
			SysFreeString(res);
			return rv;
		}


		String^ CopyIntsDeviceToDevice(IntPtr destPtr, IntPtr srcPtr, unsigned int arraySize)
		{
			BSTR res = DllCopyDeviceToDevice(destPtr.ToPointer(), srcPtr.ToPointer(), sizeof(int) * arraySize);
			String^ rv = gcnew String(res);
			SysFreeString(res);
			return rv;
		}


		////////////////////////////////////////////
		/// Floats
		////////////////////////////////////////////

		String^ MallocFloatsOnDevice(IntPtr %devPtr, unsigned int arraySize)
		{
			void *dPtr;
			BSTR res = DllMallocOnDevice(&dPtr, sizeof(float) * arraySize);
			devPtr = IntPtr(dPtr);
			String^ rv = gcnew String(res);
			SysFreeString(res);
			return rv;
		}

		String^ CopyFloatsToDevice(cli::array<float>^ data, IntPtr devPtr, unsigned int arraySize)
		{
			float *hostPtr = (float *)malloc(sizeof(float) * arraySize);
			for (unsigned int i = 0; i < arraySize; i++)
			{
				hostPtr[i] = data[i];
			}

			BSTR res = DllCopyToDevice(devPtr.ToPointer(), hostPtr, sizeof(float) * arraySize);
			free(hostPtr);
			String^ rv = gcnew String(res);
			SysFreeString(res);
			return rv;
		}

		String^ CopyFloatsFromDevice(cli::array<float>^ retlist, IntPtr devPtr, unsigned int arraySize)
		{
			float *hostPtr = (float *)malloc(sizeof(float) * arraySize);
			BSTR res = DllCopyFromDevice(hostPtr, devPtr.ToPointer(), sizeof(float) * arraySize);

			for (unsigned int i = 0; i < arraySize; i++)
			{
				retlist[i] = hostPtr[i];
			}
			free(hostPtr);

			String^ rv = gcnew String(res);
			SysFreeString(res);
			return rv;
		}

		String^ CopyFloatsDeviceToDevice(IntPtr destPtr, IntPtr srcPtr, unsigned int arraySize)
		{
			BSTR res = DllCopyDeviceToDevice(destPtr.ToPointer(), srcPtr.ToPointer(), sizeof(float) * arraySize);
			String^ rv = gcnew String(res);
			SysFreeString(res);
			return rv;
		}


		////////////////////////////////////////////
		/// Doubles
		////////////////////////////////////////////

		String^ MallocDoublesOnDevice(IntPtr %devPtr, unsigned int arraySize)
		{
			void *dPtr;
			BSTR res = DllMallocOnDevice(&dPtr, sizeof(double) * arraySize);
			devPtr = IntPtr(dPtr);
			String^ rv = gcnew String(res);
			SysFreeString(res);
			return rv;
		}

		String^ CopyDoublesToDevice(cli::array<double>^ data, IntPtr devPtr, unsigned int arraySize)
		{
			double *hostPtr = (double *)malloc(sizeof(double) * arraySize);
			for (unsigned int i = 0; i < arraySize; i++)
			{
				hostPtr[i] = data[i];
			}

			BSTR res = DllCopyToDevice(devPtr.ToPointer(), hostPtr, sizeof(double) * arraySize);
			free(hostPtr);
			String^ rv = gcnew String(res);
			SysFreeString(res);
			return rv;
		}

		String^ CopyDoublesFromDevice(cli::array<double>^ retlist, IntPtr devPtr, unsigned int arraySize)
		{
			double *hostPtr = (double *)malloc(sizeof(double) * arraySize);
			BSTR res = DllCopyFromDevice(hostPtr, devPtr.ToPointer(), sizeof(double) * arraySize);

			for (unsigned int i = 0; i < arraySize; i++)
			{
				retlist[i] = hostPtr[i];
			}
			free(hostPtr);
			String^ rv = gcnew String(res);
			delete res;
			return rv;
		}

		String^ CopyDoublesDeviceToDevice(IntPtr destPtr, IntPtr srcPtr, unsigned int arraySize)
		{
			BSTR res = DllCopyDeviceToDevice(destPtr.ToPointer(), srcPtr.ToPointer(), sizeof(double) * arraySize);
			String^ rv = gcnew String(res);
			SysFreeString(res);
			return rv;
		}


		////////////////////////////////////////////
		/// Block reductions
		////////////////////////////////////////////

		String^ RunLinearAddIntsKernel(IntPtr destPtr, IntPtr srcPtr, unsigned int length_in, unsigned int length_out)
		{
			BSTR res = DllRunLinearAddIntsKernel(
				(int *)destPtr.ToPointer(),
				(const int *)srcPtr.ToPointer(),
				length_in,
				length_out
			);
			String^ rv = gcnew String(res);
			SysFreeString(res);
			return rv;
		}

		String^ RunBlockAddInts_32_Kernel(IntPtr destPtr, IntPtr srcPtr, unsigned int span)
		{
			BSTR res = DllRunBlockAddInts_32_Kernel(
				(int *)destPtr.ToPointer(),
				(int *)srcPtr.ToPointer(),
				span);
			String^ rv = gcnew String(res);
			SysFreeString(res);
			return rv;
		}

		String^ RunBlockAddInts_16_Kernel(IntPtr destPtr, IntPtr srcPtr, unsigned int span)
		{
			BSTR res = DllRunBlockAddInts_16_Kernel(
				(int *)destPtr.ToPointer(),
				(int *)srcPtr.ToPointer(),
				span);
			String^ rv = gcnew String(res);
			SysFreeString(res);
			return rv;
		}

		String^ RunLinearAddFloatsKernel(IntPtr destPtr, IntPtr srcPtr, unsigned int length_in, unsigned int length_out)
		{
			BSTR res = DllRunLinearAddFloatsKernel(
				(float *)destPtr.ToPointer(),
				(const float *)srcPtr.ToPointer(),
				length_in,
				length_out
			);
			String^ rv = gcnew String(res);
			SysFreeString(res);
			return rv;
		}

		String^ RunBlockAddFloats_32_Kernel(IntPtr destPtr, IntPtr srcPtr, unsigned int span)
		{
			BSTR res = DllRunBlockAddFloats_32_Kernel(
				(float *)destPtr.ToPointer(),
				(const float *)srcPtr.ToPointer(),
				span);
			String^ rv = gcnew String(res);
			SysFreeString(res);
			return rv;
		}

		String^ RunBlockAddFloats_16_Kernel(IntPtr destPtr, IntPtr srcPtr, unsigned int span)
		{
			BSTR res = DllRunBlockAddFloats_16_Kernel(
				(float *)destPtr.ToPointer(),
				(const float *)srcPtr.ToPointer(),
				span);
			String^ rv = gcnew String(res);
			SysFreeString(res);
			return rv;
		}




		////////////////////////////////////////////
		/// System
		////////////////////////////////////////////

		String^ ReleaseDevicePtr(IntPtr devPtr)
		{
			void *dPtr = devPtr.ToPointer();
			BSTR res = DllReleaseDevicePtr(dPtr);
			String^ rv = gcnew String(res);
			SysFreeString(res);
			return rv;
		}

		String^ ResetDevice()
		{
			BSTR res = DllResetDevice();
			String^ rv = gcnew String(res);
			SysFreeString(res);
			return rv;
		}


		////////////////////////////////////////////
		/// Testing
		////////////////////////////////////////////

		String^ TestRuntimeErr()
		{
			BSTR res = DllTestRuntimeErr();
			String^ rv = gcnew String(res);
			SysFreeString(res);
			return rv;
		}

		String^ TestCudaStatusErr()
		{
			BSTR res = DllTestCudaStatusErr();
			String^ rv = gcnew String(res);
			SysFreeString(res);
			return rv;
		}

	};
}
