#pragma once
#include <wtypes.h>

extern "C" __declspec(dllexport) BSTR DllRunGolK(int *dev_out, int *dev_in, unsigned int span);

extern "C" __declspec(dllexport) BSTR DllRunCa9fK(float *output, float *input, float *rands, unsigned int span, float step_size, float noise);

extern "C" __declspec(dllexport) BSTR DllRunAltKernel(int *data, unsigned int span, int alt, int value);

extern "C" __declspec(dllexport) BSTR DllRunAltKernelCopy(int *dataOut, int *dataIn, unsigned int span, int alt, int value);

extern "C" __declspec(dllexport) BSTR DllRunAltIsingKernel(int *data, float *rands, float temp, unsigned int span, int alt);

extern "C" __declspec(dllexport) BSTR DllRunAltIsingKernelCopy(int *dataOut, int *dataIn, float *rands, float temp, unsigned int span, int alt);

extern "C" __declspec(dllexport) BSTR DllRundevice_function_init_YK(double d_t, int* d_spin, int* d_bond, double* d_random_data, unsigned int* d_label);

extern "C" __declspec(dllexport) BSTR DllRunIsingKernel(int *data, float *rands, unsigned int span, int alt, float t1, float t2, float t3, float t4);