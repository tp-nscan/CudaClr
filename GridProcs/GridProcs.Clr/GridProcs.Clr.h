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

	};
}
