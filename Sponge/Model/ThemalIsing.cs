using CuArrayClr;
using GridProcsClr;
using RandoClr;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;

namespace Sponge.Model
{
    public class ThemalIsing
    {
        private static IntPtr d_gridA;
        private static IntPtr d_gridB;
        private static uint _span;
        private static uint _area;
        private static CudaArray _cudaArray;
        private static GridProcs _gridProcs;
        private static RandoProcs _randoProcs;
        private static int _phase;
        private static Stopwatch _stopwatch = new Stopwatch();

        public static string Init(float[] inputs, uint span)
        {
            _span = span;
            _area = _span * _span;
            
            d_gridA = new IntPtr();
            d_gridB = new IntPtr();

            _cudaArray = new CudaArray();
            _gridProcs = new GridProcs();
            _randoProcs = new RandoProcs();

            var strRet = _cudaArray.ResetDevice();
            strRet = strRet + _cudaArray.MallocFloatsOnDevice(ref d_gridA, _area);
            strRet = strRet + _cudaArray.MallocFloatsOnDevice(ref d_gridB, _area);
            strRet = strRet + _cudaArray.CopyFloatsToDevice(inputs, d_gridA, _area);
            strRet = strRet + _cudaArray.CopyFloatsToDevice(inputs, d_gridB, _area);

            return strRet;
        }


    }
}
