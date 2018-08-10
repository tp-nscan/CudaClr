#pragma once
#include "stdafx.h"

extern "C" __declspec(dllimport) BSTR DllMallocOnDevice(void **dev_ints, unsigned int bytes);

extern "C" __declspec(dllimport) BSTR DllReleaseDevicePtr(void *dev_ptr);

extern "C" __declspec(dllimport) BSTR DllCopyToDevice(void *dev_ints, const void *host_ints, unsigned int bytes);

extern "C" __declspec(dllimport) BSTR DllCopyFromDevice(void *host_ints, const void *dev_ints, unsigned int bytes);

extern "C" __declspec(dllimport) BSTR DllCopyDeviceToDevice(void *dest_ints, const void *src_ints, unsigned int bytes);

extern "C" __declspec(dllimport) BSTR DllRunLinearReduceKernel(int *d_out, const int *d_in, unsigned int length_in, unsigned int length_out);

extern "C" __declspec(dllimport) BSTR DllRunBlockReduce_16_Kernel(int *d_out, const int *d_in, unsigned int span);

extern "C" __declspec(dllimport) BSTR DllRunBlockReduce_32_Kernel(int *d_out, const int *d_in, unsigned int span);

extern "C" __declspec(dllimport) BSTR DllDeviceSynchronize();

extern "C" __declspec(dllimport) BSTR DllResetDevice();

extern "C" __declspec(dllimport) BSTR DllTestRuntimeErr();

extern "C" __declspec(dllimport) BSTR DllTestCudaStatusErr();
