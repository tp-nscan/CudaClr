#pragma once
#include "GridProcs.h"

using namespace System;

namespace GridProcsClr 
{
	public ref class GridProcs
	{
	public:

		GridProcs() {};

		String^ Runk_Gol(IntPtr destPtr, IntPtr srcPtr, unsigned int span)
		{
			BSTR res = DllRun_k_Gol((int *)destPtr.ToPointer(), (int *)srcPtr.ToPointer(), span);
			String^ rv = gcnew String(res);
			SysFreeString(res);
			return rv;
		}


		String^ Runk_Energy4(IntPtr energyPtr, IntPtr srcPtr, unsigned int span)
		{
			BSTR res = DllRun_k_Energy4(
				(int *)energyPtr.ToPointer(),
				(int *)srcPtr.ToPointer(),
				span);
			String^ rv = gcnew String(res);
			SysFreeString(res);
			return rv;
		}


		String^ Runk_Ising_dg(IntPtr destPtr, IntPtr energyPtr, IntPtr srcPtr, IntPtr randPtr, unsigned int span, int alt, IntPtr threshPtr)
		{
			BSTR res = DllRun_k_Ising_dg(
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

		String^ Run_k_Ising_bp(IntPtr destPtr, IntPtr energyPtr, IntPtr indexRandPtr, IntPtr tempRandPtr, unsigned int block_size, unsigned int blocks_per_span, IntPtr threshPtr)
		{
			BSTR res = DllRun_k_Ising_bp(
				(int *)destPtr.ToPointer(),
				(int *)energyPtr.ToPointer(),
				(unsigned int *)indexRandPtr.ToPointer(),
				(float *)tempRandPtr.ToPointer(),
				block_size,
				blocks_per_span,
				(float *)threshPtr.ToPointer());
			String^ rv = gcnew String(res);
			SysFreeString(res);
			return rv;
		}

		String^ Run_k_Thermo_dg(IntPtr dataOut, IntPtr dataIn, unsigned int span, int alt, float rate, unsigned int fixed_colA, unsigned int fixed_colB)
		{
			BSTR res = DllRun_k_Thermo_dg(
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

		String^ Run_k_Thermo_bp(IntPtr dataOut, IntPtr index_rands, unsigned int block_size, unsigned int blocks_per_span, float rate, unsigned int fixed_colA, unsigned int fixed_colB)
		{
			BSTR res = DllRun_k_Thermo_bp(
				(float *)dataOut.ToPointer(),
				(unsigned int *)index_rands.ToPointer(),
				block_size,
				blocks_per_span,
				rate,
				fixed_colA,
				fixed_colB
			);
			String^ rv = gcnew String(res);
			SysFreeString(res);
			return rv;
		}


		String^ Run_k_ThermoIsing_bp(IntPtr temp_data, IntPtr flip_data, IntPtr index_rands, IntPtr flip_rands, IntPtr threshes,
									 float flip_energy, unsigned int block_size, unsigned int blocks_per_span, float q_rate)
		{
			BSTR res = DllRun_k_ThermoIsing_bp(
				(float *)temp_data.ToPointer(),
				(int *)flip_data.ToPointer(),
				(unsigned int *)index_rands.ToPointer(),
				(float *)flip_rands.ToPointer(),
				(float *)threshes.ToPointer(),
				flip_energy,
				block_size,
				blocks_per_span,
				q_rate
			);
			String^ rv = gcnew String(res);
			SysFreeString(res);
			return rv;
		}

	};
}
