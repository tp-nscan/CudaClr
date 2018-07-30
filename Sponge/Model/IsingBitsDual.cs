using System;
using System.Linq;
using CuArrayClr;
using GridProcsClr;
using RandoClr;
using System.Diagnostics;
using System.Collections.Generic;

namespace Sponge.Model
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
            d_energy = new IntPtr();
            d_energyBlocks = new IntPtr();

            _cudaArray = new CudaArray();
            _gridProcs = new GridProcs();
            _randoProcs = new RandoProcs();

            var strRet = _cudaArray.ResetDevice();
            strRet = strRet + _randoProcs.MakeGenerator64(SEED);
            strRet = strRet + _cudaArray.MallocIntsOnDevice(ref d_gridA, _area);
            strRet = strRet + _cudaArray.MallocIntsOnDevice(ref d_energy, _area);
            strRet = strRet + _cudaArray.MallocIntsOnDevice(ref d_energyBlocks, _area/1024);
            strRet = strRet + _cudaArray.MallocIntsOnDevice(ref d_gridB, _area);
            strRet = strRet + _cudaArray.CopyIntsToDevice(inputs, d_gridA, _area);
            strRet = strRet + _cudaArray.CopyIntsToDevice(inputs, d_gridB, _area);
            strRet = strRet + _cudaArray.MallocFloatsOnDevice(ref d_rands, _area);

            return strRet;
        }

        public static ProcResult UpdateMetro(int steps, float temp)
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

            var dRet = new Dictionary<string, object>();
            dRet["Grid"] = new SimGrid<int>(name: "UpdateMetro",
                                            width: _span,
                                            height: _span,
                                            data: res);
            return new ProcResult(data: dRet,
                                   err: strRet,
                                   steps: 0,
                                   time: _stopwatch.ElapsedMilliseconds);
        }


        public static ProcResult GetEnergy()
        {
            var strRet = String.Empty;

            int[] res = new int[_area/1024];

            strRet = strRet + _gridProcs.RunBlockReduce_32_Kernel(
                                    destPtr: d_energyBlocks,
                                    srcPtr: d_energy,
                                    span: _span
                );
            
            strRet = strRet + _cudaArray.CopyIntsFromDevice(res, d_energyBlocks, _area/1024);
            
            float tot = res.Sum();
            tot /= _area;

            var dRet = new Dictionary<string, object>();
            dRet["Grid"] = res;
            dRet["Energy"] = tot;
            return new ProcResult(data: dRet,
                                   err: strRet,
                                   steps: 0,
                                   time: _stopwatch.ElapsedMilliseconds);
        }

        public static ProcResult UpdateG(int steps, float temp)
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

                strRet = strRet + _gridProcs.RunIsingKernelPlusEnergy(
                        destPtr: dDest,
                        energyPtr: d_energy,
                        srcPtr: dSrc,
                        randPtr: d_rands,
                        span: _span,
                        alt: _phase,
                        t2: (float)t2,
                        t4: (float)t4
                    );
            }

            int[] res = new int[_area];
            strRet = strRet + _cudaArray.CopyIntsFromDevice(res, dDest, _area);

            _stopwatch.Stop();


            var dRet = new Dictionary<string, object>();
            dRet["Grid"] = new SimGrid<int>(name: "UpdateMetro",
                                            width: _span,
                                            height: _span,
                                            data: res);
            return new ProcResult(data: dRet,
                                   err: strRet,
                                   steps: steps,
                                   time: _stopwatch.ElapsedMilliseconds);


        }

        public static ProcResult UpdateE(int steps, float temp)
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

                strRet = strRet + _gridProcs.RunIsingKernelPlusEnergy(
                        destPtr: dDest,
                        energyPtr: d_energy,
                        srcPtr: dSrc,
                        randPtr: d_rands,
                        span: _span,
                        alt: _phase,
                        t2: (float)t2,
                        t4: (float)t4
                    );
            }

            int[] res = new int[_area];
            strRet = strRet + _cudaArray.CopyIntsFromDevice(res, d_energy, _area);

            float tot = res.Sum();
            tot /= _area;

            _stopwatch.Stop();

            var dRet = new Dictionary<string, object>();
            dRet["Grid"] = new SimGrid<int>(name: "UpdateE",
                                            width: _span,
                                            height: _span,
                                            data: res);
            dRet["Energy"] = tot;
            return new ProcResult(data: dRet,
                                   err: strRet,
                                   steps: steps,
                                   time: _stopwatch.ElapsedMilliseconds);

        }


        public static ProcResult UpdateE2(int steps, float temp, bool flip)
        {
            double t2 = (1.0 / (1.0 + Math.Exp(2 * temp)));
            double t4 = (1.0 / (1.0 + Math.Exp(4 * temp)));

            var strRet = String.Empty;
            IntPtr dSrc;
            IntPtr dDest = IntPtr.Zero;
            _stopwatch.Reset();
            _stopwatch.Start();


            float energyTot = 0;
            int[] resB = new int[_area / 1024];

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

                strRet = strRet + _gridProcs.RunIsingKernelVariableTemp(
                        destPtr: dDest,
                        energyPtr: d_energy,
                        srcPtr: dSrc,
                        randPtr: d_rands,
                        span: _span,
                        alt: _phase,
                        t2: (float)t2,
                        t4: (float)t4,
                        flip: flip
                    );

                strRet = strRet + _gridProcs.RunBlockReduce_32_Kernel(
                        destPtr: d_energyBlocks,
                        srcPtr: d_energy,
                        span: _span
                );

                strRet = strRet + _cudaArray.CopyIntsFromDevice(resB, d_energyBlocks, _area / 1024);

                energyTot += (float)resB.Sum() /(float) _area;

            }

            energyTot /= steps;

            int[] res = new int[_area];
            strRet = strRet + _cudaArray.CopyIntsFromDevice(res, dDest, _area);
            

            _stopwatch.Stop();

            var dRet = new Dictionary<string, object>();
            dRet["Grid"] = new SimGrid<int>(name: "UpdateE",
                                            width: _span,
                                            height: _span,
                                            data: res);
            dRet["Energy"] = energyTot;
            return new ProcResult(data: dRet,
                                   err: strRet,
                                   steps: steps,
                                   time: _stopwatch.ElapsedMilliseconds);
        }

    }
}
