#pragma once
#include <wtypes.h>

extern "C" __declspec(dllexport) BSTR DllRunGolKernel(int *dev_out, int *dev_in, unsigned int span);

extern "C" __declspec(dllexport) BSTR DllRunAltKernel(int *data, unsigned int span, int alt, int value);

extern "C" __declspec(dllexport) BSTR DllRunAltKernelCopy(int *dataOut, int *dataIn, unsigned int span, int alt, int value);

extern "C" __declspec(dllexport) BSTR DllRunMetroIsingKernel(int *data, float *rands, float temp, unsigned int span, int alt);

extern "C" __declspec(dllexport) BSTR DllRunMetroIsingKernelCopy(int *dataOut, int *dataIn, float *rands, float temp, unsigned int span, int alt);

extern "C" __declspec(dllexport) BSTR DllRunIsingKernel(int *dataOut, int *dataIn, float *rands, unsigned int span, int alt, float t2, float t4);

extern "C" __declspec(dllexport) BSTR DllRunIsingKernelPlusEnergy(int *dataOut, int *energyOut, int *dataIn, float *rands, unsigned int span, int alt, float t2, float t4);

extern "C" __declspec(dllexport) BSTR DllRunLinearReduceKernel(int *d_out, const int *d_in, unsigned int length_in, unsigned int length_out);

extern "C" __declspec(dllexport) BSTR DllRunBlockReduce_16_Kernel(int *d_out, const int *d_in, unsigned int span);

extern "C" __declspec(dllexport) BSTR DllRunBlockReduce_32_Kernel(int *d_out, const int *d_in, unsigned int span);

extern "C" __declspec(dllexport) BSTR DllRundevice_function_init_YK(double d_t, int* d_spin, int* d_bond, double* d_random_data, unsigned int* d_label);



