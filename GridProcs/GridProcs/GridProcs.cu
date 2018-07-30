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
		unsigned int t = sqrt(span);
		dim3 b = dim3(t, t);
		GolKernel <<<b, b>>>(dev_out, dev_in, span);
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
		unsigned int t = sqrt(span);
		dim3 b = dim3(t, t);
		AltKernel<<<b, b>>>(data, span, alt, value);
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
		unsigned int t = sqrt(span);
		dim3 b = dim3(t, t);
		AltKernelCopy<<<b, b>>>(dataOut, dataIn, span, alt, value);
		return BSTR();
	}
	catch (std::runtime_error &e)
	{
		std::string err = e.what();
		return RuntimeErrBSTR(err, funcName);
	}
}


extern "C" __declspec(dllexport) BSTR DllRunMetroIsingKernel(int *data, float *rands, float temp, unsigned int span, int alt)
{
	std::string funcName = "DllRunAltIsingKernel";
	try
	{
		unsigned int t = sqrt(span);
		dim3 b = dim3(t, t);
		MetroIsingKernel<<<b, b>>>(data, rands, temp, span, alt);
		return BSTR();
	}
	catch (std::runtime_error &e)
	{
		std::string err = e.what();
		return RuntimeErrBSTR(err, funcName);
	}
}


extern "C" __declspec(dllexport) BSTR DllRunMetroIsingKernelCopy(int *dataOut, int *dataIn, float *rands, float temp, unsigned int span, int alt)
{
	std::string funcName = "DllRunMetroIsingKernelCopy";
	try
	{
		unsigned int t = sqrt(span);
		dim3 b = dim3(t, t);
		MetroIsingKernelCopy<<<b, b>>>(dataOut, dataIn, rands, temp, span, alt);
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


extern "C" __declspec(dllexport) BSTR DllRunIsingKernelPlusEnergy(int *dataOut, int *energyOut, int *dataIn, float *rands, unsigned int span, int alt, float t2, float t4)
{
	std::string funcName = "DllRunIsingKernelPlusEnergy";
	try
	{
		int td = SqrtPow2Lb(span);
		dim3 t = dim3(td, td);
		dim3 b = dim3(span / td, span / td);
		IsingKernelPlusEnergy <<<b, t>>>(dataOut, energyOut, dataIn, rands, span, alt, t2, t4);
		return BSTR();
	}
	catch (std::runtime_error &e)
	{
		std::string err = e.what();
		return RuntimeErrBSTR(err, funcName);
	}
}


extern "C" __declspec(dllexport) BSTR DllRunIsingKernelVariableTemp(int *dataOut, int *energyOut, int *dataIn, float *rands, unsigned int span, int alt, float t2, float t4, bool flip)
{
	std::string funcName = "DllRunIsingKernelVariableTemp";
	try
	{
		int td = SqrtPow2Lb(span);
		dim3 t = dim3(td, td);
		dim3 b = dim3(span / td, span / td);
		IsingKernelVariableTemp<< <b, t >> >(dataOut, energyOut, dataIn, rands, span, alt, t2, t4, flip);
		return BSTR();
	}
	catch (std::runtime_error &e)
	{
		std::string err = e.what();
		return RuntimeErrBSTR(err, funcName);
	}
}


extern "C" __declspec(dllexport) BSTR DllRunLinearReduceKernel(int *d_out, const int *d_in, unsigned int length_in, unsigned int length_out)
{
	std::string funcName = "DllRunLinearReduceKernel";
	try
	{
		LinearReduceKernel<<<length_out, 512>>>(d_out, d_in, length_in);
		return BSTR();
	}
	catch (std::runtime_error &e)
	{
		std::string err = e.what();
		return RuntimeErrBSTR(err, funcName);
	}
}


extern "C" __declspec(dllexport) BSTR DllRunBlockReduce_16_Kernel(int *d_out, const int *d_in, unsigned int span)
{
	std::string funcName = "DllRunBlockReduce_16";
	try
	{
		dim3 b = dim3(span / 16, span / 16);
		dim3 t = dim3(16, 16);
		BlockReduce_16_Kernel <<<b, t>>>(d_out, d_in);
		return BSTR();
	}
	catch (std::runtime_error &e)
	{
		std::string err = e.what();
		return RuntimeErrBSTR(err, funcName);
	}
}


extern "C" __declspec(dllexport) BSTR DllRunBlockReduce_32_Kernel(int *d_out, const int *d_in, unsigned int span)
{
	std::string funcName = "DllRunBlockReduce_32";
	try
	{
		dim3 b = dim3(span/32, span/32);
		dim3 t = dim3(32, 32);
		BlockReduce_32_Kernel <<<b, t >> >(d_out, d_in);

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
		device_function_init_YK << <b, b >> >(d_t, d_spin, d_bond, d_random_data, d_label);
		return BSTR();
	}
	catch (std::runtime_error &e)
	{
		std::string err = e.what();
		return RuntimeErrBSTR(err, funcName);
	}
}

