#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void k_Gol(int *out, int *in, unsigned int span);

__global__ void k_Energy4(int *energyOut, int *dataIn, unsigned int span);

__global__ void k_Ising_dg(int *dataOut, int *energyOut, int *dataIn, float *rands, unsigned int span, int alt, float *thresh);

__global__ void k_RandBlockPick(int *dataOut, unsigned int *rands, unsigned int block_bits);

__global__ void k_Ising_bp(int *dataOut, int *energyOut, unsigned int *index_rands, float *temp_rands, unsigned int block_size, float *tts);

__global__ void device_function_init_YK(double d_t, int* d_spin, int* d_bond, double* d_random_data, unsigned int* d_label);

__global__ void k_Thermo_dg(float *dataOut, float *dataIn, unsigned int span, int alt, float rate, int fixed_colA, int fixed_colB);

__global__ void k_Thermo_bp(float *dataOut, unsigned int *index_rands, unsigned int block_size, float rate, int fixed_colA, int fixed_colB);

__global__ void k_ThermoIsing_bp(float *temp_data, int *flip_data, unsigned int *index_rands, float *flip_rands,
	float *threshes, float flip_energy, unsigned int block_size, float q_rate, int fixed_colA, int fixed_colB);