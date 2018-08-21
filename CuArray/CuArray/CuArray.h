#pragma once
#include <wtypes.h>

extern "C" __declspec(dllexport) BSTR DllMallocOnDevice(void **dev_ints, unsigned int bytes);

extern "C" __declspec(dllexport) BSTR DllReleaseDevicePtr(void *dev_ptr);

extern "C" __declspec(dllexport) BSTR DllCopyToDevice(void *dev_ints, const void *host_ints, unsigned int bytes);

extern "C" __declspec(dllexport) BSTR DllCopyFromDevice(void *host_ints, const void *dev_ints, unsigned int bytes);

extern "C" __declspec(dllexport) BSTR DllCopyDeviceToDevice(void *dest_ints, const void *src_ints, unsigned int bytes);

extern "C" __declspec(dllexport) BSTR DllRunLinearAddIntsKernel(int *d_out, const int *d_in, unsigned int length_in, unsigned int length_out);

extern "C" __declspec(dllexport) BSTR DllRunBlockAddInts_16_Kernel(int *d_out, const int *d_in, unsigned int span);

extern "C" __declspec(dllexport) BSTR DllRunBlockAddInts_32_Kernel(int *d_out, const int *d_in, unsigned int span);

extern "C" __declspec(dllexport) BSTR DllRunLinearAddFloatsKernel(float *d_out, const float *d_in, unsigned int length_in, unsigned int length_out);

extern "C" __declspec(dllexport) BSTR DllRunBlockAddFloats_16_Kernel(float *d_out, const float *d_in, unsigned int span);

extern "C" __declspec(dllexport) BSTR DllRunBlockAddFloats_32_Kernel(float *d_out, const float *d_in, unsigned int span);

extern "C" __declspec(dllexport) BSTR DllDeviceSynchronize();

extern "C" __declspec(dllexport) BSTR DllResetDevice();

extern "C" __declspec(dllexport) BSTR DllTestRuntimeErr();

extern "C" __declspec(dllexport) BSTR DllTestCudaStatusErr();