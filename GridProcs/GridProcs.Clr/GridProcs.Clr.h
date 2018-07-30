#pragma once
#include "GridProcs.h"

using namespace System;

namespace GridProcsClr 
{
	public ref class GridProcs
	{
	public:

		GridProcs() {};

		String^ RunGolKernel(IntPtr destPtr, IntPtr srcPtr, unsigned int span)
		{
			BSTR res = DllRunGolKernel((int *)destPtr.ToPointer(), (int *)srcPtr.ToPointer(), span);
			String^ rv = gcnew String(res);
			SysFreeString(res);
			return rv;
		}


		String^ RunAltKernel(IntPtr dataPtr, unsigned int span, int alt, int value)
		{
			BSTR res = DllRunAltKernel(
					(int *)dataPtr.ToPointer(),
					span,
					alt,
				    value);
			String^ rv = gcnew String(res);
			SysFreeString(res);
			return rv;
		}


		String^ RunAltKernelCopy(IntPtr destPtr, IntPtr srcPtr, unsigned int span, int alt, int value)
		{
			BSTR res = DllRunAltKernelCopy(
				(int *)destPtr.ToPointer(),
				(int *)srcPtr.ToPointer(),
				span,
				alt,
				value);
			String^ rv = gcnew String(res);
			SysFreeString(res);
			return rv;
		}


		String^ RunMetroIsingKernel(IntPtr dataPtr, IntPtr randPtr, float temp, unsigned int span, int alt)
		{
			BSTR res = DllRunMetroIsingKernel(
							(int *)dataPtr.ToPointer(),
							(float *)randPtr.ToPointer(),
							temp,
							span,
							alt);
			String^ rv = gcnew String(res);
			SysFreeString(res);
			return rv;
		}


		String^ RunMetroIsingKernelCopy(IntPtr destPtr, IntPtr srcPtr, IntPtr randPtr, float temp, unsigned int span, int alt)
		{
			BSTR res = DllRunMetroIsingKernelCopy(
				(int *)destPtr.ToPointer(),
				(int *)srcPtr.ToPointer(),
				(float *)randPtr.ToPointer(),
				temp,
				span,
				alt);
			String^ rv = gcnew String(res);
			SysFreeString(res);
			return rv;
		}


		String^ RunIsingKernel(IntPtr destPtr, IntPtr srcPtr, IntPtr randPtr, unsigned int span, int alt, float t2, float t4)
		{
			BSTR res = DllRunIsingKernel(
				(int *)destPtr.ToPointer(),
				(int *)srcPtr.ToPointer(),
				(float *)randPtr.ToPointer(),
				span,
				alt, t2, t4);
			String^ rv = gcnew String(res);
			SysFreeString(res);
			return rv;
		}

		String^ RunIsingKernelPlusEnergy(IntPtr destPtr, IntPtr energyPtr, IntPtr srcPtr, IntPtr randPtr, unsigned int span, int alt, float t2, float t4)
		{
			BSTR res = DllRunIsingKernelPlusEnergy(
				(int *)destPtr.ToPointer(),
				(int *)energyPtr.ToPointer(),
				(int *)srcPtr.ToPointer(),
				(float *)randPtr.ToPointer(),
				span,
				alt, t2, t4);
			String^ rv = gcnew String(res);
			SysFreeString(res);
			return rv;
		}

		String^ RunIsingKernelVariableTemp(IntPtr destPtr, IntPtr energyPtr, IntPtr srcPtr, IntPtr randPtr, unsigned int span, int alt, float t2, float t4, bool flip)
		{
			BSTR res = DllRunIsingKernelVariableTemp(
				(int *)destPtr.ToPointer(),
				(int *)energyPtr.ToPointer(),
				(int *)srcPtr.ToPointer(),
				(float *)randPtr.ToPointer(),
				span,
				alt, t2, t4, flip);
			String^ rv = gcnew String(res);
			SysFreeString(res);
			return rv;
		}

		String^ RunBlockReduce_32_Kernel(IntPtr destPtr, IntPtr srcPtr, unsigned int span)
		{
			BSTR res = DllRunBlockReduce_32_Kernel(
				(int *)destPtr.ToPointer(),
				(int *)srcPtr.ToPointer(),
				span);
			String^ rv = gcnew String(res);
			SysFreeString(res);
			return rv;
		}

		String^ RunBlockReduce_16_Kernel(IntPtr destPtr, IntPtr srcPtr, unsigned int span)
		{
			BSTR res = DllRunBlockReduce_16_Kernel(
				(int *)destPtr.ToPointer(),
				(int *)srcPtr.ToPointer(),
				span);
			String^ rv = gcnew String(res);
			SysFreeString(res);
			return rv;
		}


	};
}
