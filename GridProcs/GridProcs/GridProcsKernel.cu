#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include "GridProcsKernel.h"


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


__global__ void MetroIsingKernel(int *dataOut, int *dataIn, float *rands, float temp, unsigned int span, int alt)
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

			int goob = 0;
			int res = 0;
			if (q < 0)
			{
				res = -c;
			}
			else if ((q == 0) && (rr < 0.5))
			{
				res = -c;
			}
			else if ((q == 2) && (rr < t2))
			{
				res = -c;
			}
			else if ((q == 4) && (rr < t4))
			{
				res = -c;
			}
			else
			{
				res = c;
			}

			goob = res;

			dataOut[offset] = res;
		}
	}
}


__global__ void IsingKernelEnergy(int *energyOut, int *dataIn, unsigned int span)
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
		}
	}
}


__global__ void IsingKernelPlusEnergy(int *dataOut, int *energyOut, int *dataIn, float *rands, unsigned int span, int alt, float *tts)
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
			int phase = (i + j + alt) % 2;

			float rr = rands[offset];
			float thresh = tts[q + 4 + phase];

			int res = c;
			if (rr < thresh)
			{
				res = c * (-1);
			}
			dataOut[offset] = res;
		}
	}
}


__global__ void k_RandBlockPick(int *dataOut, unsigned int *rands, unsigned int block_size)
{
	unsigned int blocks_per_span = gridDim.x * blockDim.x;
	unsigned int span = blocks_per_span * block_size;
	unsigned int block_row = threadIdx.y + blockIdx.y * blockDim.y;
	unsigned int block_col = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int indexIn = block_col + block_row * blocks_per_span;

	unsigned int block_mask = block_size - 1;
	unsigned int randy = rands[indexIn] -1  ;
	unsigned int randy2 = randy >> 8;
	unsigned int in_block_row = randy & block_mask;
	unsigned int in_block_col = randy2 & block_mask;

	dataOut[in_block_col + block_col * block_size + (in_block_row + block_row * block_size) * span] += 1;
}


__global__ void k_IsingRb(int *dataOut, int *energyOut, unsigned int *index_rands, float *temp_rands, unsigned int block_size, float t2, float t4)
{
	unsigned int blocks_per_span = gridDim.x * blockDim.x;
	unsigned int span = blocks_per_span * block_size;
	unsigned int block_row = threadIdx.y + blockIdx.y * blockDim.y;
	unsigned int block_col = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int indexIn = block_col + block_row * blocks_per_span;

	unsigned int block_mask = block_size - 1;
	unsigned int randy = index_rands[indexIn];
	unsigned int randy2 = randy >> 8;
	unsigned int in_block_row = randy & block_mask;
	unsigned int in_block_col = randy2 & block_mask;

	unsigned int row = in_block_row + block_row * block_size;
	unsigned int col = in_block_col + block_col * block_size;
	unsigned int index = col + row * span;


	int c = dataOut[index];

	unsigned int row_m = (row - 1 + span) % span;
	unsigned int row_p = (row + 1) % span;
	unsigned int col_m = (col - 1 + span) % span;
	unsigned int col_p = (col + 1) % span;

	unsigned int top = dataOut[row_m * span + col];
	unsigned int l = dataOut[row * span + col_m];
	unsigned int r = dataOut[row * span + col_p];
	unsigned int bot = dataOut[row_p * span + col];

	int q = (top + l + r + bot) * c;
	energyOut[index] = q;

	float rr = temp_rands[indexIn];

	int res = 0;

	if (rr > .5)
	{
		res = -1;
	}
	else {

		if (q == -4)
		{
			res = (rr > t4) ? -c : c;
		}
		else if (q == -2)
		{
			res = (rr > t2) ? -c : c;
		}
		else if (q == 0)
		{
			res = (rr < 0.5) ? -c : c;
		}
		else if (q == 2)
		{
			res = (rr < t2) ? -c : c;
		}
		else if (q == 4)
		{
			res = (rr < t4) ? -c : c;
		}
	}
	////if ((q == -4) && (rr > t4))
	////{
	////	res = -c;
	////}
	////else if ((q == -2) && (rr > t2))
	////{
	////	res = -c;
	////}
	//if (q < 0)
	//{
	//	res = -c;
	//}
	//else if ((q = 0) && (rr < 0.5))
	//{
	//	res = -c;
	//}
	//else if ((q = 2) && (rr < t2))
	//{
	//	res = -c;
	//}
	//else if ((q = 4) && (rr < t4))
	//{
	//	res = -c;
	//}
	dataOut[index] = res;
}


__global__ void k_IsingRb8(int *dataOut, int *energyOut, unsigned int *index_rands, float *temp_rands, unsigned int block_size, float t2, float t4)
{
	unsigned int blocks_per_span = gridDim.x * blockDim.x;
	unsigned int span = blocks_per_span * block_size;
	unsigned int block_row = threadIdx.y + blockIdx.y * blockDim.y;
	unsigned int block_col = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int indexIn = block_col + block_row * blocks_per_span;

	unsigned int block_mask = block_size - 1;
	unsigned int randy = index_rands[indexIn];
	unsigned int randy2 = randy >> 8;
	unsigned int in_block_row = randy & block_mask;
	unsigned int in_block_col = randy2 & block_mask;

	unsigned int row = in_block_row + block_row * block_size;
	unsigned int col = in_block_col + block_col * block_size;
	unsigned int index = col + row * span;


	int c = dataOut[index];

	unsigned int row_m = (row - 1 + span) % span;
	unsigned int row_p = (row + 1) % span;
	unsigned int col_m = (col - 1 + span) % span;
	unsigned int col_p = (col + 1) % span;

	unsigned int row_m2 = (row - 2 + span) % span;
	unsigned int row_p2 = (row + 2) % span;
	unsigned int col_m2 = (col - 2 + span) % span;
	unsigned int col_p2 = (col + 2) % span;

	unsigned int top = dataOut[row_m * span + col];
	unsigned int l = dataOut[row * span + col_m];
	unsigned int r = dataOut[row * span + col_p];
	unsigned int bot = dataOut[row_p * span + col];

	int q = (top + l + r + bot) * c;
	energyOut[index] = q;

	float rr = temp_rands[indexIn];

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
	dataOut[index] = res;
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


__global__ void k_Thermo(float *dataOut, float *dataIn, unsigned int span, int alt, float rate)
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

			float top = dataIn[im * span + j];
			float l = dataIn[i * span + jm];
			float r = dataIn[i * span + jp];
			float bot = dataIn[ip * span + j];

			float q = c + (top + l + r + bot) * rate;

			if (q < -1.0) q = -1.0;
			if (q > 1.0) q = 1.0;
			dataOut[offset] = q;
		}
	}
}

