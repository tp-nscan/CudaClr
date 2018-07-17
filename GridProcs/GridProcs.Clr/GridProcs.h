#pragma once
#include "stdafx.h"

extern "C" __declspec(dllimport) BSTR DllRunGolK(int *dev_out, const int *dev_in, int span);

extern "C" __declspec(dllimport) BSTR DllRunCa9fK(float *output, const float *input, float *rands, const int span, float step_size, float noise);
