#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include "GridProcsKernel.h"


//int main()
//{
//    const int arraySize = 5;
//    const int a[arraySize] = { 1, 2, 3, 4, 5 };
//    const int b[arraySize] = { 10, 20, 30, 40, 50 };
//    int c[arraySize] = { 0 };
//
//    // Add vectors in parallel.
//    cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "addWithCuda failed!");
//        return 1;
//    }
//
//    printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
//        c[0], c[1], c[2], c[3], c[4]);
//
//    // cudaDeviceReset must be called before exiting in order for profiling and
//    // tracing tools such as Nsight and Visual Profiler to show complete traces.
//    cudaStatus = cudaDeviceReset();
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaDeviceReset failed!");
//        return 1;
//    }
//
//    return 0;
//}


__global__ void GolKernel(int *output, int *input, unsigned int span)
{
	for (int i = threadIdx.y + blockDim.y*blockIdx.y; i < span; i += gridDim.y*blockDim.y)
	{
		for (int j = threadIdx.x + blockIdx.x*blockDim.x; j < span; j += blockDim.x*gridDim.x)
		{
			int offset = i * span + j;

			int im = (i - 1 + span) % span;
			int ip = (i + 1) % span;
			int jm = (j - 1 + span) % span;
			int jp = (j + 1) % span;

			int topl = input[im * span + jm];
			int top = input[im * span + j];
			int topr = input[im * span + jp];
			int l = input[i * span + jm];
			int c = input[offset];
			int r = input[i * span + jp];
			int botl = input[ip * span + jm];
			int bot = input[ip * span + j];
			int botr = input[ip * span + jp];

			int sum = topl + top + topr + l + r + botl + bot + botr;

			if (c == 0)
			{
				output[offset] = (sum == 3) ? 1 : 0;
			}
			else
			{
				output[offset] = ((sum == 2) || (sum == 3)) ? 1 : 0;
			}
		}
	}
}


__global__ void AltKernel(int *data, unsigned int span, int alt, int value)
{
	for (int i = threadIdx.y + blockDim.y*blockIdx.y; i < span; i += gridDim.y*blockDim.y)
	{
		int tw = (i + alt) % 2;

		for (int j = threadIdx.x * 2 + tw + blockIdx.x*blockDim.x; j < span; j += blockDim.x*gridDim.x)
		{
			int offset = i * span + j;
			data[offset] = value;
		}
	}
}


__global__ void AltKernelCopy(int *dataOut, int *dataIn, unsigned int span, int alt, int value)
{
	for (int i = threadIdx.y + blockDim.y*blockIdx.y; i < span; i += gridDim.y*blockDim.y)
	{
		int tw = (i + alt) % 2;

		for (int j = threadIdx.x  + blockIdx.x*blockDim.x; j < span; j += blockDim.x*gridDim.x)
		{
			int offset = i * span + j;

			if ((j + tw) % 2)
			{
				dataOut[offset] = dataIn[offset];
            }
			else
			{
				dataOut[offset] = 1; //value;
			}
		}
	}
}


__global__ void MetroIsingKernel(int *data, float *rands, float temp, unsigned int span, int alt)
{
	for (int i = threadIdx.y + blockDim.y*blockIdx.y; i < span; i += gridDim.y*blockDim.y)
	{
		int tw = (i + alt) % 2;

		for (int j = threadIdx.x * 2 + tw + blockIdx.x*blockDim.x; j < span; j += blockDim.x*gridDim.x)
		{
			int offset = i * span + j;

			int im = (i - 1 + span) % span;
			int ip = (i + 1) % span;
			int jm = (j - 1 + span) % span;
			int jp = (j + 1) % span;

			int top = data[im * span + j];
			int l = data[i * span + jm];
			int r = data[i * span + jp];
			int bot = data[ip * span + j];

			int q = (top + l + r + bot);
			float tot = q + rands[i] * temp;
			data[offset] = (tot > 0) ? 1 : -1;
		}
	}
}


__global__ void MetroIsingKernelCopy(int *dataOut, int *dataIn, float *rands, float temp, unsigned int span, int alt)
{
	for (int i = threadIdx.y + blockDim.y*blockIdx.y; i < span; i += gridDim.y*blockDim.y)
	{
		int tw = (i + alt) % 2;

		for (int j = threadIdx.x + blockIdx.x*blockDim.x; j < span; j += blockDim.x*gridDim.x)
		{
			int offset = i * span + j;

			if ((j + tw) % 2)
			{
				dataOut[offset] = dataIn[offset];
			}
			else
			{
				int offset = i * span + j;

				int im = (i - 1 + span) % span;
				int ip = (i + 1) % span;
				int jm = (j - 1 + span) % span;
				int jp = (j + 1) % span;

				int top = dataIn[im * span + j];
				int l = dataIn[i * span + jm];
				int r = dataIn[i * span + jp];
				int bot = dataIn[ip * span + j];

				int q = (top + l + r + bot);
				float tot = q + rands[i] * temp;
				dataOut[offset] = (tot > 0) ? 1 : -1;
			}
		}
	}
}


__global__ void IsingKernel(int *dataOut, int *dataIn, float *rands, unsigned int span, int alt, float t2, float t4)
{
	for (int i = threadIdx.y + blockDim.y*blockIdx.y; i < span; i += gridDim.y*blockDim.y)
	{
		for (int j = threadIdx.x + blockIdx.x*blockDim.x; j < span; j += blockDim.x*gridDim.x)
		{
			int offset = i * span + j;
			int c = dataIn[offset];

			if (((i + j + alt) % 2) == 0)
			{
				dataOut[offset] = c;
				return;
			}

			int im = (i - 1 + span) % span;
			int ip = (i + 1) % span;
			int jm = (j - 1 + span) % span;
			int jp = (j + 1) % span;

			int top = dataIn[im * span + j];
			int l = dataIn[i * span + jm];
			int r = dataIn[i * span + jp];
			int bot = dataIn[ip * span + j];


			int q = (top + l + r + bot) * c;
			float rr = rands[offset];

			int res = c;
			if (q < 0)
			{
				res = -c;
			}
			else if ((q = 0) && (rr < 0.5))
			{
				res = -c;
			}
			else if ((q = 2) && (rr < t2))
			{
				res = -c;
			}
			else if ((q = 4) && (rr < t4))
			{
				res = -c;
			}
			dataOut[offset] = res;
		}
	}
}


__global__ void IsingKernelPlusEnergy(int *dataOut, int *energyOut, int *dataIn, float *rands, unsigned int span, int alt, float t2, float t4)
{
	for (int i = threadIdx.y + blockDim.y*blockIdx.y; i < span; i += gridDim.y*blockDim.y)
	{
		for (int j = threadIdx.x + blockIdx.x*blockDim.x; j < span; j += blockDim.x*gridDim.x)
		{
			int offset = i * span + j;
			int c = dataIn[offset];

			int im = (i - 1 + span) % span;
			int ip = (i + 1) % span;
			int jm = (j - 1 + span) % span;
			int jp = (j + 1) % span;

			int top = dataIn[im * span + j];
			int l = dataIn[i * span + jm];
			int r = dataIn[i * span + jp];
			int bot = dataIn[ip * span + j];


			int q = (top + l + r + bot) * c;
			energyOut[offset] = q;

			if (((i + j + alt) % 2) == 0)
			{
				dataOut[offset] = c;
				return;
			}

			float rr = rands[offset];

			int res = c;
			if (q < 0)
			{
				res = -c;
			}
			else if ((q = 0) && (rr < 0.5))
			{
				res = -c;
			}
			else if ((q = 2) && (rr < t2))
			{
				res = -c;
			}
			else if ((q = 4) && (rr < t4))
			{
				res = -c;
			}
			dataOut[offset] = res;
		}
	}
}


__global__ void IsingKernelVariableTemp(int *dataOut, int *energyOut, int *dataIn, float *rands, unsigned int span, int alt, float t2, float t4, bool flip)
{
	float temp2 = t2;
	float temp4 = t4;
	if (flip)
	{
		temp2 = t4;
		temp4 = t2;
	}

	for (int i = threadIdx.y + blockDim.y*blockIdx.y; i < span; i += gridDim.y*blockDim.y)
	{
		for (int j = threadIdx.x + blockIdx.x*blockDim.x; j < span; j += blockDim.x*gridDim.x)
		{
			int offset = i * span + j;
			int c = dataIn[offset];

			int im = (i - 1 + span) % span;
			int ip = (i + 1) % span;
			int jm = (j - 1 + span) % span;
			int jp = (j + 1) % span;

			int top = dataIn[im * span + j];
			int l = dataIn[i * span + jm];
			int r = dataIn[i * span + jp];
			int bot = dataIn[ip * span + j];


			int q = (top + l + r + bot) * c;
			energyOut[offset] = q;

			if (((i + j + alt) % 2) == 0)
			{
				dataOut[offset] = c;
				return;
			}

			float rr = rands[offset];

			int res = c;
			if (q < 0)
			{
				res = -c;
			}
			else if ((q = 0) && (rr < 0.5))
			{
				res = -c;
			}
			else if ((q = 2) && (rr < temp2))
			{
				res = -c;
			}
			else if ((q = 4) && (rr < temp4))
			{
				res = -c;
			}
			dataOut[offset] = res;
		}
	}
}


__global__ void LinearReduceKernel(int *d_out, const int *d_in, unsigned int length)
{
	__shared__ float cache[512];

	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int cacheIndex = threadIdx.x;

	int cumer = 0;
	while (tid < length)
	{
		cumer += d_in[tid];
		tid += blockDim.x * gridDim.x;
	}
	cache[cacheIndex] = cumer;

	__syncthreads();

	int i = blockDim.x / 2;
	while (i != 0)
	{
		if (cacheIndex < i)
			cache[cacheIndex] += cache[cacheIndex + i];
		__syncthreads();
		i /= 2;
	}

	if (cacheIndex == 0)
	{
		for (int i = 0; i < blockDim.x; i++)
		{
			d_out[blockIdx.x] = cache[0];
		}
	}
}


__global__ void BlockReduce_32_Kernel(int *d_out, const int *d_in)
{
	__shared__ float cache[1024];

	int cacheIndex = threadIdx.y + threadIdx.x * blockDim.x;
	int span = gridDim.x * blockDim.x;
	int rowIn = threadIdx.y + blockIdx.y * blockDim.y;
	int colIn = threadIdx.x + blockIdx.x * blockDim.x;
	int indexIn = rowIn + colIn * span;
	int indexOut = blockIdx.y + blockIdx.x * gridDim.x;

	cache[cacheIndex] = d_in[indexIn];

	__syncthreads();

	int i = 512;
	while (i != 0)
	{
		if (cacheIndex < i)
			cache[cacheIndex] += cache[cacheIndex + i];
		__syncthreads();
		i /= 2;
	}

	if (cacheIndex == 0)
	{
		d_out[indexOut] = cache[0];
	}
}


__global__ void BlockReduce_16_Kernel(int *d_out, const int *d_in)
{
	__shared__ float cache[256];

	int cacheIndex = threadIdx.y + threadIdx.x * blockDim.x;
	int span = gridDim.x * blockDim.x;
	int rowIn = threadIdx.y + blockIdx.y * blockDim.y;
	int colIn = threadIdx.x + blockIdx.x * blockDim.x;
	int indexIn = rowIn + colIn * span;
	int indexOut = blockIdx.y + blockIdx.x * gridDim.x;

	cache[cacheIndex] = d_in[indexIn];

	__syncthreads();

	int i = 128;
	while (i != 0)
	{
		if (cacheIndex < i)
			cache[cacheIndex] += cache[cacheIndex + i];
		__syncthreads();
		i /= 2;
	}

	if (cacheIndex == 0)
	{
		d_out[indexOut] = cache[0];
	}
}


//****************************************************************************
#define gSpan 16  // linear system size
const int gArea = gSpan * gSpan;


__global__ void device_function_init_YK(double d_t, int* d_spin,
	int* d_bond, double* d_random_data, unsigned int* d_label)
	/*
	Bond connection
	(Komura algorithm)
	*/
{
	unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
	int la, i, index_min;
	int spin, bond;
	__shared__ double boltz;

	spin = d_spin[index];
	bond = 0;
	index_min = index;
	if (threadIdx.x == 0) {
		boltz = d_t;
	}
	__syncthreads();

	/*------------ Bond connections with left and top sites ---------------*/

	for (i = 0; i<2; i++) {
		if (i == 0)la = (index - 1 + gSpan) % gSpan + ((int)(index / gSpan))*gSpan;
		if (i == 1)la = (index - gSpan + gArea) % gArea;
		if (spin == d_spin[la]) {
			if (boltz < d_random_data[index + i * gArea]) {
				bond |= 0x01 << i;
				index_min = min(index_min, la);
			}
		}
	}

	/*------------ Transfer to global memories ----------------------------*/

	// Transfer "label" and "bond" to a global memory
	d_bond[index] = bond;
	d_label[index] = index_min;
}
