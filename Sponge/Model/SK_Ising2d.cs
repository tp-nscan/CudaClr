using System;
using System.Collections.Generic;
using CuArrayClr;
using GridProcsClr;
using RandoClr;
using Utils;

namespace Sponge.Model
{
    public class SK_Ising2d
    {
        private static CudaArray _cudaArray;
        private static GridProcs _gridProcs;
        private static RandoProcs _randoProcs;

        private const int SEED = 123;
        private const uint MAXTHREADS = 512;

        private static IntPtr d_rands;
        private static IntPtr d_grid;

        private static uint _span;
        private static uint _area;

        private static int _phase;
        private static float _temp = 2.0f;

        private static uint _mem_N;
        private static uint _mem_rand;
        private static uint _mem_1 = sizeof(int) * (1);
        private static uint _mem_measured_quantity;
        private static uint _mem_measured_magnet;

        private static uint _blockSize;
        private static uint _gridSize;


        public static string Init(int[] inputs, uint span)
        {
            // init libs
            _cudaArray = new CudaArray();
            _gridProcs = new GridProcs();
            _randoProcs = new RandoProcs();

            // Set grid sizes
            _span = span;
            _area = _span * _span;

            // Set block and thread sizes
            _blockSize = (_span < MAXTHREADS) ? _span : MAXTHREADS;
            _gridSize = _area / _blockSize;

            // Set memory sizes
            _mem_N = sizeof(int) * (_area);
            _mem_rand = sizeof(double) * (3 * _area);
            _mem_1 = sizeof(int) * (1);
            _mem_measured_quantity = sizeof(int) * (_gridSize);
            _mem_measured_magnet = sizeof(int) * (_gridSize);

            // Allocate device arrays
            d_rands = new IntPtr();
            d_grid = new IntPtr();

            var strRet = _cudaArray.ResetDevice();
            strRet = strRet + _randoProcs.MakeGenerator64(SEED);
            strRet = strRet + _cudaArray.MallocIntsOnDevice(ref d_grid, _area);
            strRet = strRet + _cudaArray.CopyIntsToDevice(inputs, d_grid, _area);
            strRet = strRet + _cudaArray.MallocFloatsOnDevice(ref d_rands, _area);

            return strRet;
        }


        public static ProcResult Update(int[] results, int steps)
        {
            var strRet = String.Empty;


            var dRet = new Dictionary<string, object>();

            //dRet["Grid"] = new SimGrid<int>(name: "UpdateMetro",
            //                                width: _span,
            //                                height: _span,
            //                                data: res);
            return new ProcResult(data: dRet,
                                   err: strRet,
                                   steps: steps,
                                   time: 0);
        }





    }
}
