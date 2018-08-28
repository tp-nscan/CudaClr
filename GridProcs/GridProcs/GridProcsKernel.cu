#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include "GridProcsKernel.h"
#include <math.h>


__global__ void k_Gol(int *output, int *input, unsigned int span)
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


__global__ void k_Energy4(int *energyOut, int *dataIn, unsigned int span)
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


__global__ void k_Ising_dg(int *dataOut, int *energyOut, int *dataIn, float *rands, unsigned int span, int alt, float *tts)
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


__global__ void k_Ising_bp(int *flip_data, int *energyOut, unsigned int *index_rands, float *temp_rands, unsigned int block_size, float *tts)
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

	int c = flip_data[index];

	unsigned int row_m = (row - 1 + span) % span;
	unsigned int row_p = (row + 1) % span;
	unsigned int col_m = (col - 1 + span) % span;
	unsigned int col_p = (col + 1) % span;

	int top = flip_data[row_m * span + col];
	int l = flip_data[row * span + col_m];
	int r = flip_data[row * span + col_p];
	int bot = flip_data[row_p * span + col];

	int q = (top + l + r + bot) * c;
	energyOut[index] = q;

	float rr = temp_rands[indexIn];

	float thresh = tts[q / 2 + 2];

	int res = c;
	if (rr < thresh)
	{
		res = c * (-1);
	}
	flip_data[index] = res;
}


__global__ void k_Ising_bp8(int *dataOut, int *energyOut, unsigned int *index_rands, float *temp_rands, unsigned int block_size, float t2, float t4)
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
//****************************************************************************
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


__global__ void k_Thermo_dg(float *dataOut, float *dataIn, unsigned int span, int alt, float rate, int fixed_colA, int fixed_colB)
{
	for (int i = threadIdx.y + blockDim.y*blockIdx.y; i < span; i += gridDim.y*blockDim.y)
	{
		for (int j = threadIdx.x + blockIdx.x*blockDim.x; j < span; j += blockDim.x*gridDim.x)
		{
			int offset = i * span + j;
			float c = dataIn[offset];

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

			float q = c + (top + l + r + bot - 4*c) * rate;

			if (q < 0.0) q = 0.0;
			if (q > 1.0) q = 1.0;

			if ((j == fixed_colA) || (j == fixed_colB))
			{
				dataOut[offset] = c;
			}
			else
			{
				dataOut[offset] = q;
			}
		}
	}
}


__global__ void k_Thermo_bp(float *grid_data, unsigned int *index_rands, unsigned int block_size, float rate, int fixed_colA, int fixed_colB) 
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

	float c = grid_data[index];

	unsigned int row_m = (row - 1 + span) % span;
	unsigned int row_p = (row + 1) % span;
	unsigned int col_m = (col - 1 + span) % span;
	unsigned int col_p = (col + 1) % span;

	float top = grid_data[row_m * span + col];
	float l = grid_data[row * span + col_m];
	float r = grid_data[row * span + col_p];
	float bot = grid_data[row_p * span + col];

	float q = c + (top + l + r + bot - 4 * c) * rate;

	if (q < 0.0) q = 0.0;
	if (q > 1.0) q = 1.0;

	if ((col == fixed_colA) || (col == fixed_colB))
	{
		grid_data[index] = c;
	}
	else
	{
		grid_data[index] = q;
	}

}


__global__ void k_ThermoIsing_bp(float *temp_data, int *flip_data, unsigned int *index_rands, float *flip_rands,
	float *threshes, float flip_energy, unsigned int block_size, float q_rate)
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

	float c_t = temp_data[index];
	int c_f = flip_data[index];

	unsigned int row_m = (row - 1 + span) % span;
	unsigned int row_p = (row + 1) % span;
	unsigned int col_m = (col - 1 + span) % span;
	unsigned int col_p = (col + 1) % span;

	unsigned int t_dex = row_m * span + col;
	unsigned int l_dex = row * span + col_m;
	unsigned int r_dex = row * span + col_p;
	unsigned int b_dex = row_p * span + col;

	float top_t = temp_data[t_dex];
	float l_t = temp_data[l_dex];
	float r_t = temp_data[r_dex];
	float bot_t = temp_data[b_dex];

	int top_f = flip_data[t_dex];
	int l_f = flip_data[l_dex];
	int r_f = flip_data[r_dex];
	int bot_f = flip_data[b_dex];

	int q_f = (top_f + l_f + r_f + bot_f) * c_f;
	float flip_t = 0;
	
	float rr = flip_rands[indexIn];
	int threshDex = q_f * (1 - c_t) * 32.0 + 128;
	float thresh = threshes[threshDex];
	int res_f = c_f;

	if (rr < thresh)
	{
		res_f = c_f * (-1);
		flip_t = flip_energy;
		flip_t *= q_f;
	}
	else
	{
		flip_t = 0;
	}

	flip_data[index] = res_f;

	//float d_top_t = top_t - c_t;
	//float d_l_t = l_t - c_t;
	//float d_r_t = r_t - c_t;
	//float d_bot_t = bot_t - c_t;

	//float d2_top_t = d_top_t * d_top_t;
	//float d2_l_t = d_l_t * d_l_t;
	//float d2_r_t = d_r_t * d_r_t;
	//float d2_bot_t = d_bot_t * d_bot_t;

	//float res_t = c_t + (
	//		d2_top_t * fabsf(d_top_t) + 
	//		d2_l_t * fabsf(d_l_t) +
	//		d2_r_t * fabsf(d_r_t) +
	//		d2_bot_t * fabsf(d_bot_t)
	//	) * q_rate - flip_t;

	float res_t = c_t + (top_t + l_t + r_t + bot_t - 4 * c_t) * q_rate - flip_t;

	if (res_t < 0.0) res_t = 0.0;
	if (res_t > 0.99) res_t = 0.99;

	temp_data[index] = res_t;
}

 
__global__ void k_ThermoIsing_bp0(float *temp_data, int *flip_data, unsigned int *index_rands, float *flip_rands,
	float *threshes, float flip_energy, unsigned int block_size, float q_rate, int fixed_colA, int fixed_colB)
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

	float c_t = temp_data[index];
	int c_f = flip_data[index];

	unsigned int row_m = (row - 1 + span) % span;
	unsigned int row_p = (row + 1) % span;
	unsigned int col_m = (col - 1 + span) % span;
	unsigned int col_p = (col + 1) % span;

	unsigned int t_dex = row_m * span + col;
	unsigned int l_dex = row * span + col_m;
	unsigned int r_dex = row * span + col_p;
	unsigned int b_dex = row_p * span + col;

	float top_t = temp_data[t_dex];
	float l_t = temp_data[l_dex];
	float r_t = temp_data[r_dex];
	float bot_t = temp_data[b_dex];

	int top_f = flip_data[t_dex];
	int l_f = flip_data[l_dex];
	int r_f = flip_data[r_dex];
	int bot_f = flip_data[b_dex];

	int q_f = (top_f + l_f + r_f + bot_f) * c_f;
	float flip_t = 0;

	float rr = flip_rands[indexIn];
	int threshDex = q_f * (1 - c_t) * 32.0 + 128;
	float thresh = threshes[threshDex];
	int res_f = c_f;

	if (rr < thresh)
	{
		res_f = c_f * (-1);
		flip_t = flip_energy;
		flip_t *= q_f;
	}
	else
	{
		flip_t = 0;
	}

	flip_data[index] = res_f;

	float res_t = c_t + (top_t + l_t + r_t + bot_t - 4 * c_t) * q_rate - flip_t;
	if (res_t < 0.0) res_t = 0.0;
	if (res_t > 1.0) res_t = 1.0;

	if ((col == fixed_colA) || (col == fixed_colB))
	{
		temp_data[index] = c_t;
	}
	else
	{
		temp_data[index] = res_t;
	}

}

