using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using CuArrayClr;
using GridProcsClr;
using RandoClr;
using Utils;

namespace Sponge.Model.Lattice
{
    public static class BlockPick
    {
        private const int SEED = 123;
        private static IntPtr d_indexRands;
        private static IntPtr d_tempRands;
        private static IntPtr d_energy;
        private static IntPtr d_energyBlocks;
        private static int[] h_energyBlocks;
        private static IntPtr d_betas;

        private static IntPtr d_grid;
        private static uint _span;
        private static uint _block_size;
        private static uint _blocks_per_span;
        private static uint _area;
        private static uint _blockCount;
        private static CudaArray _cudaArray;
        private static GridProcs _gridProcs;
        private static RandoProcs _randoProcs;

        
        private static Stopwatch _stopwatch = new Stopwatch();


        public static string Init(int[] inputs, uint span, uint blockSize)
        {
            _span = span;
            _block_size = blockSize;

            _area = _span * _span;
            _blocks_per_span = span / blockSize;
            _blockCount = _blocks_per_span * _blocks_per_span;

            d_indexRands = new IntPtr();
            d_grid = new IntPtr();

            _cudaArray = new CudaArray();
            _gridProcs = new GridProcs();
            _randoProcs = new RandoProcs();

            var strRet = _cudaArray.ResetDevice();
            strRet = strRet + _randoProcs.MakeGenerator32(SEED);
            strRet = strRet + _cudaArray.MallocIntsOnDevice(ref d_grid, _area);
            strRet = strRet + _cudaArray.MallocIntsOnDevice(ref d_energy, _area);
            strRet = strRet + _cudaArray.MallocIntsOnDevice(ref d_energyBlocks, _area / 1024);
            strRet = strRet + _cudaArray.CopyIntsToDevice(inputs, d_grid, _area);
            strRet = strRet + _cudaArray.MallocIntsOnDevice(ref d_indexRands, _blockCount);
            strRet = strRet + _cudaArray.MallocFloatsOnDevice(ref d_tempRands, _blockCount);
            strRet = strRet + _gridProcs.Runk_Energy4(d_energy, d_grid, _span);
            strRet = strRet + _cudaArray.MallocFloatsOnDevice(ref d_betas, 5);

            return strRet;
        }


        public static ProcResult ProcMarkBlocks(int steps)
        {
            var strRet = String.Empty;
            int[] res = new int[_area];

            _stopwatch.Reset();
            _stopwatch.Start();

            for (var i = 0; i < steps; i++)
            {
                //uint[] rrands = IntArrayGen.RandUInts((int)DateTime.Now.Ticks, _blockCount);
                //strRet = strRet + _cudaArray.CopyUIntsToDevice(rrands, d_rands, _blockCount);

                strRet = strRet + _randoProcs.MakeRandomInts(d_indexRands, _blockCount);

                strRet = strRet + _gridProcs.Run_k_RandBlockPick(
                                        destPtr: d_grid,
                                        randPtr: d_indexRands,
                                        block_size: _block_size,
                                        blocks_per_span: _blocks_per_span
                    );
            }

            strRet = strRet + _cudaArray.CopyIntsFromDevice(res, d_grid, _area);

            _stopwatch.Stop();


            //uint[] res2 = new uint[_blockCount];
            //strRet = strRet + _cudaArray.CopyUIntsFromDevice(res2, d_rands, _blockCount);

            var dRet = new Dictionary<string, object>();

            dRet["Grid"] = new SimGrid<int>(name: "Update",
                                            width: _span,
                                            height: _span,
                                            data: res);
 
            return new ProcResult(data: dRet,
                                   err: strRet,
                                   stepsCompleted: steps,
                                   time: _stopwatch.ElapsedMilliseconds);
        }


        public static ProcResult ProcIsingRb(int steps, float temp)
        {
            var strRet = String.Empty;

            float t2 = (float)(1.0 / (1.0 + Math.Exp(2 * temp)));
            float t4 = (float)(1.0 / (1.0 + Math.Exp(4 * temp)));

            float[] thresh = new float[5];
            //thresh[0] = 1.0f;
            //thresh[1] = 1.0f;
            thresh[0] = 1.0f - t4;
            thresh[1] = 1.0f - t2;
            thresh[2] = 0.5f;
            thresh[3] = t2;
            thresh[4] = t4;

            strRet = strRet + _cudaArray.CopyFloatsToDevice(thresh, d_betas, 5);

            _stopwatch.Reset();
            _stopwatch.Start();

            for (var i = 0; i < steps; i++)
            {
                strRet = strRet + _randoProcs.MakeRandomInts(d_indexRands, _blockCount);
                strRet = strRet + _randoProcs.MakeUniformRands(d_tempRands, _blockCount);

                strRet = strRet + _gridProcs.Run_k_Ising_bp(
                        destPtr: d_grid,
                        energyPtr: d_energy,
                        indexRandPtr: d_indexRands,
                        tempRandPtr: d_tempRands,
                        block_size: _block_size,
                        blocks_per_span: _blocks_per_span,
                        threshPtr: d_betas
                    );
            }

            strRet = strRet + _gridProcs.Runk_Energy4(d_energy, d_grid, _span);

            int[] mres = new int[_area / 1024];

            strRet = strRet + _cudaArray.RunBlockAddInts_32_Kernel(
                                    destPtr: d_energyBlocks,
                                    srcPtr: d_energy,
                                    span: _span);

            strRet = strRet + _cudaArray.CopyIntsFromDevice(mres, d_energyBlocks, _area / 1024);
            
            float tot = mres.Sum();
            tot /= _area;


            int[] res = new int[_area];
            strRet = strRet + _cudaArray.CopyIntsFromDevice(res, d_grid, _area);

            _stopwatch.Stop();


            //uint[] res2 = new uint[_blockCount];
            //strRet = strRet + _cudaArray.CopyUIntsFromDevice(res2, d_rands, _blockCount);

            var dRet = new Dictionary<string, object>();

            dRet["Grid"] = new SimGrid<int>(name: "Update",
                                            width: _span,
                                            height: _span,
                                            data: res);

            dRet["Energy"] = tot;

            return new ProcResult(data: dRet,
                                   err: strRet,
                                   stepsCompleted: steps,
                                   time: _stopwatch.ElapsedMilliseconds);
        }




    }
}
