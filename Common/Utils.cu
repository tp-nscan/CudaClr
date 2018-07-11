#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdlib.h>
#include <stdio.h>

int *IntArray(int length, int first = 0, int step = 0)
{
	int *av = (int *)malloc(sizeof(int) * length);
	for (int i = 0; i < length; i++)
	{
		av[i] = first + step * i;
	}
	return av;
}

	return true;
}

bool CompFloatArrays(float *a, float *b, int length)
{
	for (int i = 0; i < length; i++)
	{
		if (a[i] != b[i]) return false;
	}
	return true;
}

void PrintFloatArray(float *aa, int width, int length)
{
	for (int i = 0; i < length; i++) {
		printf("%3.3f ", aa[i]);
		if ((i>0) && ((i + 1) % width == 0)) printf("\n");
	}
	printf("\n");
}

void PrintIntArray(int *aa, int width, int length)
{
	for (int i = 0; i < length; i++) {
		printf("%d ", aa[i]);
		if ((i>0) && ((i + 1) % width == 0)) printf("\n");
	}
	printf("\n");
}

float *RndFloat0to1(int arraySize)
{
	float *temp = (float*)malloc(arraySize * sizeof(float));
	for (int i = 0; i<arraySize; i++) {
		temp[i] = (float)rand() / (float)(RAND_MAX);
	}
	return temp;
}

int *Rnd0or1(int arraySize, float fracOnes)
{
	int *temp = (int*)malloc(arraySize * sizeof(float));
	for (int i = 0; i<arraySize; i++) {
		float fv = (float)rand() / (float)(RAND_MAX);
		temp[i] = fv < fracOnes ? 1 : 0;
	}
	return temp;
}
