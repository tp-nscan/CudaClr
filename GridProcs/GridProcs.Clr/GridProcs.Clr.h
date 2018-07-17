#pragma once
#include "GridProcs.h"

using namespace System;

namespace GridProcsClr {
	public ref class GridProcs
	{
	public:
		GridProcs() {};

		String^ RunGolK(IntPtr destPtr, IntPtr srcPtr, unsigned int span)
		{
			BSTR res = DllRunGolK((int *)destPtr.ToPointer(), (int *)srcPtr.ToPointer(), span);
			String^ rv = gcnew String(res);
			SysFreeString(res);
			return rv;
		}


		String^ RunCa9fK(IntPtr outPtr, IntPtr inPtr, IntPtr rands, unsigned int span, float step_size, float noise)
		{
			BSTR res = DllRunCa9fK(
				(float *)outPtr.ToPointer(),
				(float *)inPtr.ToPointer(),
				(float *)rands.ToPointer(),
				span,
				step_size,
				noise);
			String^ rv = gcnew String(res);
			SysFreeString(res);
			return rv;
		}

	};
}
