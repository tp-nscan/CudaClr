#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <wtypes.h>
#include <stdexcept>
#include <sstream>
#include <string>
#include "kernel.h"
#include "..\..\Common\ClrUtils.h"


extern "C" __declspec(dllexport) BSTR DllRunGolK(int *dev_out, const int *dev_in, int span)
{
	std::string funcName = "DllRunGolK";
	try
	{
		int threads = (span < 1024) ? span : 1024;
		int blocks = span / threads;
		GolKernel <<<blocks, threads >>>(dev_out, dev_in, span);
		return BSTR();
	}
	catch (std::runtime_error &e)
	{
		std::string err = e.what();
		return RuntimeErrBSTR(err, funcName);
	}
}


extern "C" __declspec(dllexport) BSTR DllRunCa9fK(float *output, const float *input, float *rands, const int span, float step_size, float noise)
{
	std::string funcName = "DllRunCa9fK";
	try
	{
		int threads = (span < 1024) ? span : 1024;
		int blocks = span / threads;
		Ca9fKernel <<<blocks, threads>>>(output, input, rands, span, step_size, noise);
		return BSTR();
	}
	catch (std::runtime_error &e)
	{
		std::string err = e.what();
		return RuntimeErrBSTR(err, funcName);
	}
}