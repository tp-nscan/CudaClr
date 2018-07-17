
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

__global__ void copyKernel(int *out, const int *in)
{
	int i = threadIdx.x;
	out[i] = in[i];
}

__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

__global__ void GolKernel(int *output, const int *input, const int span)
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


__global__ void Ca9iKernel(int *output, const int *input, const int span, float *rands)
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


__global__ void Ca9fKernel(float *output, const float *input, float *rands, const int span, float step_size, float noise)
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

			float topl = input[im * span + jm];
			float top = input[im * span + j];
			float topr = input[im * span + jp];
			float l = input[i * span + jm];
			float c = input[offset];
			float r = input[i * span + jp];
			float botl = input[ip * span + jm];
			float bot = input[ip * span + j];
			float botr = input[ip * span + jp];

			float sum = c +  (topl + top + topr + l + r + botl + bot + botr + rands[offset] * noise) * step_size;

			if (sum > 1.0f)
			{
				sum = 1.0f;
			}
			if (sum < -1.0f)
			{
				sum = -1.0f;
			}
			output[offset] = sum;
		}
	}
}


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

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    addKernel<<<1, size>>>(dev_c, dev_a, dev_b);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}