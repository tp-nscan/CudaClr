using System;
using System.Collections.Generic;
using System.Diagnostics;
using CuArrayClr;
using GridProcsClr;
using RandoClr;
using Utils;

namespace Sponge.Model.Lattice
{
    public static class Themal_dg
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

        public static ProcResult UpdateH(int steps, float rate)
        {
            var strRet = String.Empty;

            IntPtr dSrc;
            IntPtr dDest = IntPtr.Zero;

            _stopwatch.Reset();
            _stopwatch.Start();

           
            for (var s = 0; s < steps; s++)
            {
                if (_phase == 0)
                {
                    dSrc = d_gridA;
                    dDest = d_gridB;
                    _phase = 1;
                }
                else
                {
                    dSrc = d_gridB;
                    dDest = d_gridA;
                    _phase = 0;
                }

                strRet = strRet + _gridProcs.Run_k_Thermo_dg(
                     dataOut: dDest,
                     dataIn: dSrc,
                     span: _span,
                     alt: _phase,
                     rate: rate,
                     fixed_colA: _span - 1,
                     fixed_colB: _span / 4);
            }
            

            var res = new float[_area];
            strRet = strRet + _cudaArray.CopyFloatsFromDevice(res, dDest, _area);

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
