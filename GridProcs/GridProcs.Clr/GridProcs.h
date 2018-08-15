#pragma once
#include "stdafx.h"

extern "C" __declspec(dllimport) BSTR DllRun_k_Gol(int *dev_out, int *dev_in, unsigned int span);

extern "C" __declspec(dllimport) BSTR DllRun_k_Energy4(int *energyOut, int *dataIn, unsigned int span);

extern "C" __declspec(dllimport) BSTR DllRun_k_Ising_dg(int *dataOut, int *energyOut, int *dataIn, float *rands, unsigned int span, int alt, float *thresh);

extern "C" __declspec(dllimport) BSTR DllRun_k_RandBlockPick(int *dataOut, unsigned int *rands, unsigned int block_size, unsigned int blocks_per_span);

extern "C" __declspec(dllimport) BSTR DllRun_k_Ising_bp(int *dataOut, int *energyOut, unsigned int *index_rands, float *temp_rands, unsigned int block_size, unsigned int blocks_per_span, float *tts);

extern "C" __declspec(dllimport) BSTR DllRundevice_function_init_YK(double d_t, int* d_spin, int* d_bond, double* d_random_data, unsigned int* d_label);

extern "C" __declspec(dllimport) BSTR DllRun_k_Thermo_dg(float *dataOut, float *dataIn, unsigned int span, int alt, float rate, unsigned int fixed_colA, unsigned int fixed_colB);

extern "C" __declspec(dllimport) BSTR DllRun_k_Thermo_bp(float *dataOut, unsigned int *index_rands, unsigned int block_size, unsigned int blocks_per_span, float rate, unsigned int fixed_colA, unsigned int fixed_colB);

extern "C" __declspec(dllimport) BSTR DllRun_k_ThermoIsing_bp(float *temp_data, int *flip_data, unsigned int *index_rands, float *flip_rands, float *threshes, float flip_energy, unsigned int block_size,
															  unsigned int blocks_per_span, float q_rate, unsigned int fixed_colA, unsigned int fixed_colB);