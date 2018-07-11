#pragma once
#include "cuda_runtime.h"
#include <stdlib.h>

int *IntArray(int length, int first = 0, int step = 0);
bool CompIntArrays(int *a, int *b, int length);
bool CompFloatArrays(float *a, float *b, int length);
void PrintFloatArray(float *aa, int width, int length);
void PrintIntArray(int *aa, int width, int length);
float *RndFloat0to1(int arraySize);
int *Rnd0or1(int arraySize, float fracOnes);
