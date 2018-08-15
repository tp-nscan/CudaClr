#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <wtypes.h>
#include <stdexcept>
#include <sstream>
#include <string>
#include "GridProcsKernel.h"
#include "..\..\Common\ClrUtils.h"


extern "C" __declspec(dllexport) BSTR DllRun_k_Gol(int *dev_out, int *dev_in, int span)
{
	std::string funcName = "DllRunGolK";
	try
	{
		int td = SqrtPow2Lb(span);
		td = (td > 32) ? 32 : td;
		dim3 t = dim3(td, td);
		dim3 b = dim3(span / td, span / td);
		k_Gol <<<b, t>>>(dev_out, dev_in, span);
		return BSTR();
	}
	catch (std::runtime_error &e)
	{
		std::string err = e.what();
		return RuntimeErrBSTR(err, funcName);
	}
}

extern "C" __declspec(dllexport) BSTR DllRun_k_Ising_dg(int *dataOut, int *energyOut, int *dataIn, float *rands, unsigned int span, int alt, float *thresh)
{
	std::string funcName = "DllRun_k_Ising";
	try
	{
		int td = SqrtPow2Lb(span);
		td = (td > 32) ? 32 : td;
		dim3 t = dim3(td, td);
		dim3 b = dim3(span / td, span / td);
		k_Ising_dg<< <b, t >> >(dataOut, energyOut, dataIn, rands, span, alt, thresh);
		return BSTR();
	}
	catch (std::runtime_error &e)
	{
		std::string err = e.what();
		return RuntimeErrBSTR(err, funcName);
	}
}


extern "C" __declspec(dllexport) BSTR DllRun_k_Energy4(int *energyOut, int *dataIn, unsigned int span)
{
	std::string funcName = "DllRun_k_Energy4";
	try
	{
		int td = SqrtPow2Lb(span);
		td = (td > 32) ? 32 : td;
		dim3 t = dim3(td, td);
		dim3 b = dim3(span / td, span / td);
		k_Energy4<<<b, t>>>(energyOut, dataIn, span);
		return BSTR();
	}
	catch (std::runtime_error &e)
	{
		std::string err = e.what();
		return RuntimeErrBSTR(err, funcName);
	}
}


extern "C" __declspec(dllexport) BSTR DllRun_k_RandBlockPick(int *dataOut, unsigned int *rands, unsigned int block_size, unsigned int blocks_per_span)
{
	std::string funcName = "DllRun_k_RandBlockPick";
	try
	{
		int td = SqrtPow2Lb(blocks_per_span);
		td = (td > 32) ? 32 : td;
		dim3 t = dim3(td, td);
		dim3 b = dim3(blocks_per_span / td, blocks_per_span / td);

		k_RandBlockPick<<<b,t>>>(dataOut, rands, block_size);
		return BSTR();
	}
	catch (std::runtime_error &e)
	{
		std::string err = e.what();
		return RuntimeErrBSTR(err, funcName);
	}
}


extern "C" __declspec(dllexport) BSTR DllRun_k_Ising_bp(int *dataOut, int *energyOut, unsigned int *index_rands, 
	float *temp_rands, unsigned int block_size, unsigned int blocks_per_span, float *tts)
{
	std::string funcName = "DllRun_k_Ising_bp";
	try
	{
		int td = SqrtPow2Lb(blocks_per_span);
		td = (td > 32) ? 32 : td;
		dim3 t = dim3(td, td);
		dim3 b = dim3(blocks_per_span / td, blocks_per_span / td);

		k_Ising_bp<<<b, t>>>(dataOut, energyOut, index_rands, temp_rands, block_size, tts);
		return BSTR();
	}
	catch (std::runtime_error &e)
	{
		std::string err = e.what();
		return RuntimeErrBSTR(err, funcName);
	}
}


extern "C" __declspec(dllexport) BSTR DllRundevice_function_init_YK(double d_t, int* d_spin, int* d_bond, double* d_random_data, unsigned int* d_label)
{
	std::string funcName = "DllRundevice_function_init_YK";
	try
	{
		dim3 b = dim3(16, 16);
		device_function_init_YK<<<b, b>>>(d_t, d_spin, d_bond, d_random_data, d_label);
		return BSTR();
	}
	catch (std::runtime_error &e)
	{
		std::string err = e.what();
		return RuntimeErrBSTR(err, funcName);
	}
}


extern "C" __declspec(dllexport) BSTR DllRun_k_Thermo_dg(float *dataOut, float *dataIn, unsigned int span, int alt, float rate, unsigned int fixed_colA, unsigned int fixed_colB)
{
	std::string funcName = "DllRun_k_Thermo_dg";
	try
	{
		int td = SqrtPow2Lb(span);
		td = (td > 32) ? 32 : td;
		dim3 t = dim3(td, td);
		dim3 b = dim3(span / td, span / td);

		k_Thermo_dg<<<b, t>>>(dataOut, dataIn, span, alt, rate, fixed_colA, fixed_colB);
		return BSTR();
	}
	catch (std::runtime_error &e)
	{
		std::string err = e.what();
		return RuntimeErrBSTR(err, funcName);
	}
}


extern "C" __declspec(dllexport) BSTR DllRun_k_Thermo_bp(float *dataOut, unsigned int *index_rands, unsigned int block_size, 
								unsigned int blocks_per_span, float rate, unsigned int fixed_colA, unsigned int fixed_colB)
{
	std::string funcName = "DllRun_k_Thermo_bp";
	try
	{
		int td = SqrtPow2Lb(blocks_per_span);
		td = (td > 32) ? 32 : td;
		dim3 t = dim3(td, td);
		dim3 b = dim3(blocks_per_span / td, blocks_per_span / td);

		k_Thermo_bp <<<b, t >>>(dataOut, index_rands, block_size, rate, fixed_colA, fixed_colB);
		return BSTR();
	}
	catch (std::runtime_error &e)
	{
		std::string err = e.what();
		return RuntimeErrBSTR(err, funcName);
	}
}

extern "C" __declspec(dllexport) BSTR DllRun_k_ThermoIsing_bp(float *temp_data, int *flip_data,
	unsigned int *index_rands, float *flip_rands, float *threshes, float flip_energy, unsigned int block_size,
	unsigned int blocks_per_span, float q_rate, unsigned int fixed_colA, unsigned int fixed_colB)
{
	std::string funcName = "DllRun_k_ThermoIsing_bp";
	try
	{
		int td = SqrtPow2Lb(blocks_per_span);
		td = (td > 32) ? 32 : td;
		dim3 t = dim3(td, td);
		dim3 b = dim3(blocks_per_span / td, blocks_per_span / td);

		k_ThermoIsing_bp<<<b, t>>>(temp_data, flip_data, index_rands, flip_rands, threshes, flip_energy, block_size, q_rate, fixed_colA, fixed_colB);
		return BSTR();
	}
	catch (std::runtime_error &e)
	{
		std::string err = e.what();
		return RuntimeErrBSTR(err, funcName);
	}
}