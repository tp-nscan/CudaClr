using System;
using CuArrayClr;
using GridProcsClr;
using RandoClr;
using System.Diagnostics;

namespace Sponge.M2
{
    public class IsingBitsDual
    {
        private const int SEED = 123;
        private static IntPtr d_rands;
        private static IntPtr d_energy;
        private static IntPtr d_energyBlocks;
        private static int[] h_energyBlocks;

        private static IntPtr d_gridA;
        private static IntPtr d_gridB;
        private static uint _span;
        private static uint _area;
        private static CudaArray _cudaArray;
        private static GridProcs _gridProcs;
        private static RandoProcs _randoProcs;
        private static int _phase;
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
            strRet = strRet + _cudaArray.MallocIntsOnDevice(ref d_energy, _area);
            strRet = strRet + _cudaArray.MallocIntsOnDevice(ref d_energyBlocks, _area);
            strRet = strRet + _cudaArray.MallocIntsOnDevice(ref d_gridB, _area);
            strRet = strRet + _cudaArray.CopyIntsToDevice(inputs, d_gridA, _area);
            strRet = strRet + _cudaArray.CopyIntsToDevice(inputs, d_gridB, _area);
            strRet = strRet + _cudaArray.MallocFloatsOnDevice(ref d_rands, _area);

            return strRet;
        }

        public static ProcResult<SimGrid<int>> UpdateMetro(int steps, float temp)
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

                strRet = strRet + _gridProcs.RunMetroIsingKernelCopy(destPtr:dDest, srcPtr: dSrc, randPtr:d_rands, temp:temp, span: _span, alt: _phase);
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


        public static ProcResult<SimGrid<int>> UpdateG(int steps, float temp)
        {
            double t2 = (1.0 / (1.0 + Math.Exp(2 * temp)));
            double t4 = (1.0 / (1.0 + Math.Exp(4 * temp)));

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
                    strRet = strRet + _randoProcs.MakeUniformRands(d_rands, _area);
                    _phase = 1;
                }
                else
                {
                    dSrc = d_gridB;
                    dDest = d_gridA;
                    _phase = 0;
                }

                strRet = strRet + _gridProcs.RunIsingKernelCopy(
                    destPtr: dDest, srcPtr: dSrc, randPtr: d_rands, span: _span, alt: _phase,
                    t2: (float)t2, t4: (float)t4
                    );
            }


            strRet = strRet + _randoProcs.MakeUniformRands(d_rands, _area);

            int[] res = new int[_area];
            strRet = strRet + _cudaArray.CopyIntsFromDevice(res, dDest, _area);

            _stopwatch.Stop();

            return new ProcResult<SimGrid<int>>(data: new SimGrid<int>(name: "Update2",
                                                                      width: _span,
                                                                      height: _span,
                                                                      data: res),
                                                err: strRet,
                                                steps: steps,
                                                time: _stopwatch.ElapsedMilliseconds);
        }


    }
}
