using System;
using System.Collections.Generic;
using System.Diagnostics;
using CuArrayClr;
using GridProcsClr;
using RandoClr;
using Utils;

namespace Sponge.Model.Lattice
{
    public static class Thermal_bp
    {
        private const int SEED = 123;
        private static IntPtr d_indexRands;

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

        public static string Init(float[] inputs, uint span, uint blockSize)
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
            strRet = strRet + _cudaArray.MallocFloatsOnDevice(ref d_grid, _area);
            strRet = strRet + _cudaArray.CopyFloatsToDevice(inputs, d_grid, _area);
            strRet = strRet + _cudaArray.MallocIntsOnDevice(ref d_indexRands, _blockCount);

            return strRet;
        }

        public static ProcResult UpdateH(int steps, float rate)
        {
            var strRet = String.Empty;

            _stopwatch.Reset();
            _stopwatch.Start();

            for (var s = 0; s < steps; s++)
            {
                strRet = strRet + _randoProcs.MakeRandomInts(d_indexRands, _blockCount);

                strRet = strRet + _gridProcs.Run_k_Thermo_bp(
                        dataOut: d_grid,
                        index_rands: d_indexRands,
                        block_size: _block_size,
                        blocks_per_span: _blocks_per_span,
                        rate: rate,
                        fixed_colA: _span / 4,
                        fixed_colB: 3 * _span / 4);
            }


            var res = new float[_area];
            strRet = strRet + _cudaArray.CopyFloatsFromDevice(res, d_grid, _area);

            _stopwatch.Stop();

            var dRet = new Dictionary<string, object>();
            dRet["Grid"] = new SimGrid<float>(name: "UpdateH",
                                              width: _span,
                                              height: _span,
                                              data: res);

            return new ProcResult(data: dRet,
                                  err: strRet,
                                  stepsCompleted: steps,
                                  time: _stopwatch.ElapsedMilliseconds);
        }

    }
}
