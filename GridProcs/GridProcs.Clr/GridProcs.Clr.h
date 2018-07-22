#pragma once
#include "GridProcs.h"

using namespace System;

namespace GridProcsClr 
{
	public ref class GridProcs
	{
	public:

		GridProcs() {};

		String^ RunGolK(IntPtr destPtr, IntPtr srcPtr, unsigned int span)
		{
			BSTR res = DllRunGolK((int *)destPtr.ToPointer(), (int *)srcPtr.ToPointer(), span);
			String^ rv = gcnew String(res);
			SysFreeString(res);
			return rv;
		}


		String^ RunCa9fK(IntPtr outPtr, IntPtr inPtr, IntPtr rands, unsigned int span, float step_size, float noise)
		{
			BSTR res = DllRunCa9fK(
				(float *)outPtr.ToPointer(),
				(float *)inPtr.ToPointer(),
				(float *)rands.ToPointer(),
				span,
				step_size,
				noise);
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

		String^ RunAltIsingKernel(IntPtr dataPtr, IntPtr randPtr, float temp, unsigned int span, int alt)
		{
			BSTR res = DllRunAltIsingKernel(
							(int *)dataPtr.ToPointer(),
							(float *)randPtr.ToPointer(),
							temp,
							span,
							alt);
			String^ rv = gcnew String(res);
			SysFreeString(res);
			return rv;
		}


		String^ RunAltIsingKernelCopy(IntPtr destPtr, IntPtr srcPtr, IntPtr randPtr, float temp, unsigned int span, int alt)
		{
			BSTR res = DllRunAltIsingKernelCopy(
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


		String^ RunIsingKernel(IntPtr dataPtr, IntPtr randPtr, unsigned int span, int alt, float t1, float t2, float t3, float t4)
		{
			BSTR res = DllRunIsingKernel(
				(int *)dataPtr.ToPointer(),
				(float *)randPtr.ToPointer(),
				span,
				alt, t1, t2, t3, t4);
			String^ rv = gcnew String(res);
			SysFreeString(res);
			return rv;
		}
	};
}
