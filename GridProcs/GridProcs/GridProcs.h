#pragma once
#include <wtypes.h>

extern "C" __declspec(dllexport) BSTR DllRunGolK(int *dev_out, const int *dev_in, int span);

extern "C" __declspec(dllexport) BSTR DllRunCa9fK(float *output, const float *input, float *rands, const int span, float step_size, float noise);
