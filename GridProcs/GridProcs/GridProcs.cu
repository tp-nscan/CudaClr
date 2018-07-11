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
		GolKernel <<<span, span>>>(dev_out, dev_in, span);
		return BSTR();
	}
	catch (std::runtime_error &e)
	{
		std::string err = e.what();
		return RuntimeErrBSTR(err, funcName);
	}
}