#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <wtypes.h>
#include <stdexcept>
#include <sstream>
#include <string>
#include "kernel.h"
#include "..\..\Common\ClrUtils.h"


extern "C" __declspec(dllexport) BSTR DllRunGolK(int *dev_out, int *dev_in, int span)
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


extern "C" __declspec(dllexport) BSTR DllRunCa9fK(float *output, float *input, float *rands, int span, float step_size, float noise)
{
	std::string funcName = "DllRunCa9fK";
	try
	{
		unsigned int t = sqrt(span);
		dim3 b = dim3(t, t);
		Ca9fKernel<<<b, b>>>(output, input, rands, span, step_size, noise);
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


extern "C" __declspec(dllexport) BSTR DllRunAltIsingKernel(int *data, float *rands, float temp, unsigned int span, int alt)
{
	std::string funcName = "DllRunAltIsingKernel";
	try
	{
		unsigned int t = sqrt(span);
		dim3 b = dim3(t, t);
		AltIsingKernel<<<b, b>>>(data, rands, temp, span, alt);
		return BSTR();
	}
	catch (std::runtime_error &e)
	{
		std::string err = e.what();
		return RuntimeErrBSTR(err, funcName);
	}
}


extern "C" __declspec(dllexport) BSTR DllRunAltIsingKernelCopy
	(int *dataOut, int *dataIn, float *rands, float temp, unsigned int span, int alt)
{
	std::string funcName = "DllRunAltIsingKernelCopy";
	try
	{
		unsigned int t = sqrt(span);
		dim3 b = dim3(t, t);
		AltIsingKernelCopy<<<b, b>>>(dataOut, dataIn, rands, temp, span, alt);
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
		device_function_init_YK<<<b, b >>>(d_t, d_spin, d_bond, d_random_data, d_label);
		return BSTR();
	}
	catch (std::runtime_error &e)
	{
		std::string err = e.what();
		return RuntimeErrBSTR(err, funcName);
	}
}


extern "C" __declspec(dllexport) BSTR DllRunIsingKernel(int *data, float *rands, unsigned int span, int alt, float t1, float t2, float t3, float t4)
{
	std::string funcName = "DllRunIsingKernel";
	try
	{
		unsigned int t = sqrt(span);
		dim3 b = dim3(32, 32);
		IsingKernel<<<b, b>>>(data, rands, span, alt, t1, t2, t3, t4);
		return BSTR();
	}
	catch (std::runtime_error &e)
	{
		std::string err = e.what();
		return RuntimeErrBSTR(err, funcName);
	}
}