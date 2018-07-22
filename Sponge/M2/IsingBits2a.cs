using System;
using CuArrayClr;
using GridProcsClr;
using RandoClr;
using System.Diagnostics;

namespace Sponge.M2
{
    public class IsingBits2a
    {
        private const int SEED = 123;
        private static IntPtr d_rands;
        private static IntPtr d_gridA;
        private static IntPtr d_gridB;
        private static uint _span;
        private static uint _area;
        private static CudaArray _cudaArray;
        private static GridProcs _gridProcs;
        private static RandoProcs _randoProcs;
        private static int _phase;
        private static float _temp = 2.0f;
        private static Stopwatch _stopwatch = new Stopwatch();

        public static string Init(int[] inputs, uint span)
        {
            _span = span;
            _area = _span * _span;

            d_rands = new IntPtr();
            d_gridA = new IntPtr();
            d_gridB = new IntPtr();

            _cudaArray = new CudaArray();
            _gridProcs = new GridProcs();
            _randoProcs = new RandoProcs();

            var strRet = _cudaArray.ResetDevice();
            strRet = strRet + _randoProcs.MakeGenerator64(SEED);
            strRet = strRet + _cudaArray.MallocIntsOnDevice(ref d_gridA, _area);
            strRet = strRet + _cudaArray.MallocIntsOnDevice(ref d_gridB, _area);
            strRet = strRet + _cudaArray.CopyIntsToDevice(inputs, d_gridA, _area);
            strRet = strRet + _cudaArray.MallocFloatsOnDevice(ref d_rands, _area);

            return strRet;
        }

        public static ProcResult<SimGrid<int>> Update(int steps, float temp)
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
                    strRet = strRet + _randoProcs.MakeNormalRands(d_rands, _area, mean: 0.0f, stdev: 0.95f);
                    _phase = 1;
                }
                else
                {
                    dSrc = d_gridB;
                    dDest = d_gridA;
                    _phase = 0;
                }

                strRet = strRet + _gridProcs.RunAltIsingKernelCopy(destPtr:dDest, srcPtr: dSrc, randPtr:d_rands, temp:temp, span: _span, alt: _phase);
            }

            int[] res = new int[_area];
            strRet = strRet + _cudaArray.CopyIntsFromDevice(res, dDest, _area);

            _stopwatch.Stop();

            return new ProcResult<SimGrid<int>>(data: new SimGrid<int>(name: "Update2",
                                                                      width: _span,
                                                                      height: _span,
                                                                      data: res),
                                                err: strRet,
                                                steps: steps,
                                                time: _stopwatch.ElapsedMilliseconds / 1000.0);
        }


        //public static string Update(int[] results, int steps)
        //{
        //    var strRet = String.Empty;
        //    for (int i = 0; i < steps; i++)
        //    {
        //        strRet = _randoProcs.MakeUniformRands(d_floats, _area);
        //    }

        //    float[] res = new float[_area];
        //    strRet = _cudaArray.CopyFloatsFromDevice(res, d_floats, _area);

        //    for(var i=0; i<_area; i++)
        //    {
        //        results[i] = (res[i] > 0.5f) ? 1 : 0;
        //    }

        //    return strRet;
        //}



    }
}
