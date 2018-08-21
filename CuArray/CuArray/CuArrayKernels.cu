#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"


__global__ void LinearAddIntsKernel(int *d_out, const int *d_in, unsigned int length)
{
	__shared__ int cache[512];

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


__global__ void BlockAddInts_32_Kernel(int *d_out, const int *d_in)
{
	__shared__ int cache[1024];

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


__global__ void BlockAddInts_16_Kernel(int *d_out, const int *d_in)
{
	__shared__ int cache[256];

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


__global__ void LinearAddFloatsKernel(float *d_out, const float *d_in, unsigned int length)
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


__global__ void BlockAddFloats_32_Kernel(float *d_out, const float *d_in)
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


__global__ void BlockAddFloats_16_Kernel(float *d_out, const float *d_in)
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

