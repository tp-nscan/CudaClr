#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <wtypes.h>
#include <stdexcept>
#include <sstream>
#include <string>
#include "GridProcsKernel.h"
#include "..\..\Common\ClrUtils.h"


extern "C" __declspec(dllexport) BSTR DllRunGolKernel(int *dev_out, int *dev_in, int span)
{
	std::string funcName = "DllRunGolK";
	try
	{
		int td = SqrtPow2Lb(span);
		td = (td > 32) ? 32 : td;
		dim3 t = dim3(td, td);
		dim3 b = dim3(span / td, span / td);
		GolKernel <<<b, t>>>(dev_out, dev_in, span);
		return BSTR();
	}
	catch (std::runtime_error &e)
	{
		std::string err = e.what();
		return RuntimeErrBSTR(err, funcName);
	}
}


extern "C" __declspec(dllexport) BSTR DllRunAltKernel(int *data, unsigned int span, int alt, int value)
{
	std::string funcName = "DllRunAltKernel";
	try
	{
		int td = SqrtPow2Lb(span);
		td = (td > 32) ? 32 : td;
		dim3 t = dim3(td, td);
		dim3 b = dim3(span / td, span / td);
		AltKernel<<<b, t>>>(data, span, alt, value);
		return BSTR();
	}
	catch (std::runtime_error &e)
	{
		std::string err = e.what();
		return RuntimeErrBSTR(err, funcName);
	}
}


extern "C" __declspec(dllexport) BSTR DllRunAltKernelCopy(int *dataOut, int *dataIn, unsigned int span, int alt, int value)
{
	std::string funcName = "DllRunAltKernelCopy";
	try
	{
		int td = SqrtPow2Lb(span);
		td = (td > 32) ? 32 : td;
		dim3 t = dim3(td, td);
		dim3 b = dim3(span / td, span / td);
		AltKernelCopy<<<b, t>>>(dataOut, dataIn, span, alt, value);
		return BSTR();
	}
	catch (std::runtime_error &e)
	{
		std::string err = e.what();
		return RuntimeErrBSTR(err, funcName);
	}
}


extern "C" __declspec(dllexport) BSTR DllRunMetroIsingKernel(int *dataOut, int *dataIn, float *rands, float temp, unsigned int span, int alt)
{
	std::string funcName = "DllRunMetroIsingKernel";
	try
	{
		int td = SqrtPow2Lb(span);
		td = (td > 32) ? 32 : td;
		dim3 t = dim3(td, td);
		dim3 b = dim3(span / td, span / td);
		MetroIsingKernel<<<b, t>>>(dataOut, dataIn, rands, temp, span, alt);
		return BSTR();
	}
	catch (std::runtime_error &e)
	{
		std::string err = e.what();
		return RuntimeErrBSTR(err, funcName);
	}
}


extern "C" __declspec(dllexport) BSTR DllRunIsingKernel(int *dataOut, int *dataIn, float *rands, unsigned int span, int alt, float t2, float t4)
{
	std::string funcName = "DllRunIsingKernel";
	try
	{
		int td = SqrtPow2Lb(span);
		td = (td > 32) ? 32 : td;
		dim3 t = dim3(td, td);
		dim3 b = dim3(span / td, span / td);
		IsingKernel<<<b, t>>>(dataOut, dataIn, rands, span, alt, t2, t4);
		return BSTR();
	}
	catch (std::runtime_error &e)
	{
		std::string err = e.what();
		return RuntimeErrBSTR(err, funcName);
	}
}


extern "C" __declspec(dllexport) BSTR DllRunIsingKernelPlusEnergy(int *dataOut, int *energyOut, int *dataIn, float *rands, unsigned int span, int alt, float *thresh)
{
	std::string funcName = "DllRunIsingKernelPlusEnergy";
	try
	{
		int td = SqrtPow2Lb(span);
		td = (td > 32) ? 32 : td;
		dim3 t = dim3(td, td);
		dim3 b = dim3(span / td, span / td);
		IsingKernelPlusEnergy << <b, t >> >(dataOut, energyOut, dataIn, rands, span, alt, thresh);
		return BSTR();
	}
	catch (std::runtime_error &e)
	{
		std::string err = e.what();
		return RuntimeErrBSTR(err, funcName);
	}
}


extern "C" __declspec(dllexport) BSTR DllRunIsingKernelEnergy(int *energyOut, int *dataIn, unsigned int span)
{
	std::string funcName = "DllRunIsingKernelEnergy";
	try
	{
		int td = SqrtPow2Lb(span);
		td = (td > 32) ? 32 : td;
		dim3 t = dim3(td, td);
		dim3 b = dim3(span / td, span / td);
		IsingKernelEnergy<<<b, t>>>(energyOut, dataIn, span);
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


extern "C" __declspec(dllexport) BSTR DllRun_k_IsingRb(int *dataOut, int *energyOut, unsigned int *index_rands, float *temp_rands, unsigned int block_size, unsigned int blocks_per_span, float t2, float t4)
{
	std::string funcName = "DllRun_k_IsingRb";
	try
	{
		int td = SqrtPow2Lb(blocks_per_span);
		td = (td > 32) ? 32 : td;
		dim3 t = dim3(td, td);
		dim3 b = dim3(blocks_per_span / td, blocks_per_span / td);

		k_IsingRb<<<b, t>>>(dataOut, energyOut, index_rands, temp_rands, block_size, t2, t4);
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


extern "C" __declspec(dllexport) BSTR DllRun_k_Thermo(float *dataOut, float *dataIn, unsigned int span, int alt, float rate)
{
	std::string funcName = "DllRun_k_Thermo";
	try
	{
		int td = SqrtPow2Lb(span);
		td = (td > 32) ? 32 : td;
		dim3 t = dim3(td, td);
		dim3 b = dim3(span / td, span / td);

		k_Thermo << <b, t >> >(dataOut, dataIn, span, alt, rate);
		return BSTR();
	}
	catch (std::runtime_error &e)
	{
		std::string err = e.what();
		return RuntimeErrBSTR(err, funcName);
	}
}

