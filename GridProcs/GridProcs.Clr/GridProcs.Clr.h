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


		String^ RunMetroIsingKernel(IntPtr destPtr, IntPtr srcPtr, IntPtr randPtr, float temp, unsigned int span, int alt)
		{
			BSTR res = DllRunMetroIsingKernel(
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


		String^ RunIsingKernelEnergy(IntPtr energyPtr, IntPtr srcPtr, unsigned int span)
		{
			BSTR res = DllRunIsingKernelEnergy(
				(int *)energyPtr.ToPointer(),
				(int *)srcPtr.ToPointer(),
				span);
			String^ rv = gcnew String(res);
			SysFreeString(res);
			return rv;
		}


		String^ RunIsingKernelPlusEnergy(IntPtr destPtr, IntPtr energyPtr, IntPtr srcPtr, IntPtr randPtr, unsigned int span, int alt, IntPtr threshPtr)
		{
			BSTR res = DllRunIsingKernelPlusEnergy(
				(int *)destPtr.ToPointer(),
				(int *)energyPtr.ToPointer(),
				(int *)srcPtr.ToPointer(),
				(float *)randPtr.ToPointer(),
				span,
				alt, 
				(float *)threshPtr.ToPointer());
			String^ rv = gcnew String(res);
			SysFreeString(res);
			return rv;
		}


		String^ Run_k_RandBlockPick(IntPtr destPtr, IntPtr randPtr, unsigned int block_size, unsigned int blocks_per_span)
		{
			BSTR res = DllRun_k_RandBlockPick(
				(int *)destPtr.ToPointer(),
				(unsigned int *)randPtr.ToPointer(),
				block_size,
				blocks_per_span);
			String^ rv = gcnew String(res);
			SysFreeString(res);
			return rv;
		}

		String^ Run_k_IsingRb(IntPtr destPtr, IntPtr energyPtr, IntPtr indexRandPtr, IntPtr tempRandPtr, unsigned int block_size, unsigned int blocks_per_span, float t2, float t4)
		{
			BSTR res = DllRun_k_IsingRb(
				(int *)destPtr.ToPointer(),
				(int *)energyPtr.ToPointer(),
				(unsigned int *)indexRandPtr.ToPointer(),
				(float *)tempRandPtr.ToPointer(),
				block_size,
				blocks_per_span,
				t2,
				t4
			);
			String^ rv = gcnew String(res);
			SysFreeString(res);
			return rv;
		}


		String^ Run_k_Thermo(IntPtr dataOut, IntPtr dataIn, unsigned int span, int alt, float rate, unsigned int fixed_colA, unsigned int fixed_colB)
		{
			BSTR res = DllRun_k_Thermo(
				(float *)dataOut.ToPointer(),
				(float *)dataIn.ToPointer(),
				span,
				alt,
				rate,
				fixed_colA,
				fixed_colB
			);
			String^ rv = gcnew String(res);
			SysFreeString(res);
			return rv;
		}


	};
}
