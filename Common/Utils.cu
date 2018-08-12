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

bool CompIntArrays(int *a, int *b, int length)
{
	for (int i = 0; i < length; i++)
	{
		if (a[i] != b[i]) return false;
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

void PrintUintArray(unsigned int *aa, int width, int length)
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

unsigned int *RndInts(int arraySize)
{
	unsigned int *temp = (unsigned int*)malloc(arraySize * sizeof(int));
	for (int i = 0; i<arraySize; i++) {
		temp[i] = rand();
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

int *Rnd_m1or1(int arraySize, float fracOnes)
{
	int *temp = (int*)malloc(arraySize * sizeof(float));
	for (int i = 0; i<arraySize; i++) {
		float fv = (float)rand() / (float)(RAND_MAX);
		temp[i] = fv < fracOnes ? 1 : -1;
	}
	return temp;
}

unsigned int SqrtPow2Lb(unsigned int rhs)
{
	unsigned int fRet = 1;
	unsigned int nv = fRet;
	while (true)
	{
		nv = fRet * 2;
		if (nv * nv > rhs)
		{
			return fRet;
		}
		fRet = nv;
	}
}

float *LeftRightGradient(unsigned int span, float low_val, float high_val)
{
	float delta = (high_val - low_val) / (span / 2.0f);
	unsigned int hs = span / 2;

	float *outputs = (float*)malloc(span * span * sizeof(float));

	for (int i = 0; i < span; i++)
	{
		for (int j = 0; j < hs; j++)
		{
			int index = i * span + j;
			outputs[index] = high_val - j * delta;
		}

		for (int j = hs; j < span; j++)
		{
			int index = i * span + j;
			outputs[index] = low_val + (j - hs) * delta;
		}

	}
	return outputs;
}
