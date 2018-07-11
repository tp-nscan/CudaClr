//#include <stdio.h>
//#include <stdlib.h>
//#include "cuda_runtime.h"
//#include "device_launch_parameters.h"
//#include "kernel.h"
//#include "..\..\Common\Utils.h"
//
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
//	runGol();
//}
//
//
//int runIsingK()
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
//	copyKernel<<<1, gridVol>>>(dev_odd, dev_even);
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