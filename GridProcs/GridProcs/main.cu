#include <stdio.h>
#include <stdlib.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "..\..\Common\Utils.h"
#include "GridProcs.h"


int runBoxPick();
int runIsingK();
int runThermal();

//void randomInit(int *data, int size)
//{
//	for (int i = 0; i < size; ++i)
//		data[i] = rand();
//}
//
//int runIsingK();
//
//int runGol();
//
//////////////////////////////////////////////////////////////////////////////////
//// Program main
//////////////////////////////////////////////////////////////////////////////////
//int main(int argc, char **argv)
//{
//	runThermal();
//}

int runThermal()
{
	const int span = 16;
	int gridVol = span * span;
	float *h_grid_A_in;
	float *h_grid_B_in;
	float *host_out;
	float *dev_A;
	float *dev_B;
	cudaError_t cudaStatus;

	printf("in runThermal\n");

	h_grid_A_in = LeftRightGradient(span, 0, 1);
	h_grid_B_in = LeftRightGradient(span, 0, 1);

	printf("in array A: \n\n");
	PrintFloatArray(h_grid_A_in, span, gridVol);

	host_out = (float *)malloc(gridVol * sizeof(float));

	cudaStatus = cudaMalloc((void**)&dev_A, gridVol * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_B, gridVol * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_A, h_grid_A_in, gridVol * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_B, h_grid_B_in, gridVol * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	DllRun_k_Thermo(dev_B, dev_A, span, 1, 0.1, 0, 3);

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching copyKernel!\n", cudaStatus);
		goto Error;
	}

	cudaStatus = cudaMemcpy(host_out, dev_B, gridVol * sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	printf("out array B: \n\n");
	PrintFloatArray(host_out, span, gridVol);

Error:
	cudaFree(dev_A);
	cudaFree(dev_B);

	return cudaStatus;
}


int runBoxPick()
{
	const int span = 32;
	const int blockSize = 4;
	const int blocks_per_span = span / blockSize;
	const int blockCount = blocks_per_span * blocks_per_span;
	int gridVol = span * span;
	unsigned int *host_rands_in;
	int *host_out;
	unsigned int *dev_rands;
	int *dev_out;
	cudaError_t cudaStatus;
	
	host_rands_in = RndInts(blockCount);
	host_out = (int *)malloc(gridVol * sizeof(int));
	
	PrintUintArray(host_rands_in, blocks_per_span, blockCount);
	printf("\n\n");


	cudaStatus = cudaMalloc((void**)&dev_rands, blockCount * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	
	cudaStatus = cudaMalloc((void**)&dev_out, gridVol * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	
	cudaStatus = cudaMemcpy(dev_rands, host_rands_in, blockCount * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}


	DllRun_k_RandBlockPick(dev_out, dev_rands, blockSize, blocks_per_span);


	cudaStatus = cudaMemcpy(host_out, dev_out, gridVol * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	PrintIntArray(host_out, span, gridVol);
	printf("\n\n");

	Error:
		cudaFree(dev_rands);
		cudaFree(dev_out);
	
		return cudaStatus;
}


int runIsingK()
{
	const int span = 16;
	int gridVol = span * span;
	int *h_grid_in;
	float *h_rands_in;
	float *h_thresh_in;
	int *host_out;
	int *dev_odd;
	int *dev_even;
	int *dev_energy;
	float *dev_rands;
	float *dev_thresh;
	cudaError_t cudaStatus;

	printf("in runIsingK\n");

	h_thresh_in = (float *)malloc(10 * sizeof(float));
	h_thresh_in[1] = 1.0;
	h_thresh_in[3] = 1.0;
	h_thresh_in[5] = 0.5;
	h_thresh_in[7] = 0.2;
	h_thresh_in[9] = 0.1;

	h_grid_in = Rnd_m1or1(gridVol, 0.3);
	h_rands_in = RndFloat0to1(gridVol);

	printf("in array: \n\n");
	PrintFloatArray(h_rands_in, span, gridVol);
	printf("in rands: \n\n");
	PrintIntArray(h_grid_in, span, gridVol);

	host_out = (int *)malloc(gridVol * sizeof(int));


	cudaStatus = cudaMalloc((void**)&dev_energy, gridVol * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_thresh, 10 * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_rands, gridVol * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_odd, gridVol * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_even, gridVol * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_even, h_grid_in, gridVol * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_rands, h_rands_in, gridVol * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_thresh, h_thresh_in, 10 * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	DllRunIsingKernelPlusEnergy(dev_odd, dev_energy, dev_even, dev_rands, span, 1, dev_thresh);

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching copyKernel!\n", cudaStatus);
		goto Error;
	}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(host_out, dev_odd, gridVol * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}


	printf("out grid: \n\n");
	PrintIntArray(host_out, span, gridVol);

Error:
	cudaFree(dev_odd);
	cudaFree(dev_even);

	return cudaStatus;
}

//
//
//int runGol()
//{
//	const int span = 28;
//	int gridVol = span * span;
//	int *host_in;
//	int *host_out;
//	int *dev_odd;
//	int *dev_even;
//	cudaError_t cudaStatus;
//
//	host_in = Rnd0or1(gridVol, 0.3);
//	host_out = (int *)malloc(gridVol * sizeof(int));
//
//	PrintIntArray(host_in, span, gridVol);
//	printf("\n\n");
//
//	cudaStatus = cudaMalloc((void**)&dev_odd, gridVol * sizeof(int));
//	if (cudaStatus != cudaSuccess) {
//		fprintf(stderr, "cudaMalloc failed!");
//		goto Error;
//	}
//
//	cudaStatus = cudaMalloc((void**)&dev_even, gridVol * sizeof(int));
//	if (cudaStatus != cudaSuccess) {
//		fprintf(stderr, "cudaMalloc failed!");
//		goto Error;
//	}
//
//	cudaStatus = cudaMemcpy(dev_even, host_in, gridVol * sizeof(int), cudaMemcpyHostToDevice);
//	if (cudaStatus != cudaSuccess) {
//		fprintf(stderr, "cudaMemcpy failed!");
//		goto Error;
//	}
//
//
//	GolKernel <<<span, span>>>(dev_odd, dev_even, span);
//
//	cudaStatus = cudaDeviceSynchronize();
//	if (cudaStatus != cudaSuccess) {
//		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching copyKernel!\n", cudaStatus);
//		goto Error;
//	}
//
//	// Copy output vector from GPU buffer to host memory.
//	cudaStatus = cudaMemcpy(host_out, dev_odd, gridVol * sizeof(int), cudaMemcpyDeviceToHost);
//	if (cudaStatus != cudaSuccess) {
//		fprintf(stderr, "cudaMemcpy failed!");
//		goto Error;
//	}
//
//
//	PrintIntArray(host_out, span, gridVol);
//
//Error:
//	cudaFree(dev_odd);
//	cudaFree(dev_even);
//
//	return cudaStatus;
//}