using System;
using CuArrayClr;
using GridProcsClr;
using RandoClr;
using System.Diagnostics;

namespace Sponge.M2
{
    public class IsingBits2
    {
        private const int SEED = 123;
        private static IntPtr d_rands;
        private static IntPtr d_grid;
        private static bool _backwards;
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
            _backwards = false;

            d_rands = new IntPtr();
            d_grid = new IntPtr();

            _cudaArray = new CudaArray();
            _gridProcs = new GridProcs();
            _randoProcs = new RandoProcs();

            var strRet = _cudaArray.ResetDevice();
            strRet = strRet + _randoProcs.MakeGenerator64(SEED);
            strRet = strRet + _cudaArray.MallocIntsOnDevice(ref d_grid, _area);
            strRet = strRet + _cudaArray.CopyIntsToDevice(inputs, d_grid, _area);
            strRet = strRet + _cudaArray.MallocFloatsOnDevice(ref d_rands, _area);

            return strRet; 
        }

        public static string Update(int[] results, int steps)
        {
            var strRet = String.Empty;

            for (var s = 0; s < steps; s++)
            {
                if (_phase == 0)
                {
                    strRet = strRet + _randoProcs.MakeNormalRands(d_rands, _area, mean: 0.0f, stdev: 0.5f);
                    _phase = 1;
                }
                else { _phase = 0; }

                strRet = strRet + _gridProcs.RunAltIsingKernel(d_grid, d_rands, temp: _temp, span: _span, alt: _phase);
                // strRet = strRet + _gridProcs.RunAltKernel(d_floats, _span, _phase);
            }

            strRet = strRet + _cudaArray.CopyIntsFromDevice(results, d_rands, _area);
           // strRet = strRet + _cudaArray.CopyIntsFromDevice(results, d_grid, _area);

            if (! string.IsNullOrEmpty(strRet))
            {
                var s = "S";
            }

            return strRet;
        }


        public static ProcResult<SimGrid<int>> Update2(int steps)
        {
            var strRet = String.Empty;
            _stopwatch.Reset();
            _stopwatch.Start();

            for (var s = 0; s < steps; s++)
            {
                if (_phase == 0)
                {
                    strRet = strRet + _randoProcs.MakeNormalRands(d_rands, _area, mean: 0.0f, stdev: 0.5f);
                    _phase = 1;
                }
                else { _phase = 0; }

                strRet = strRet + _gridProcs.RunAltIsingKernel(d_grid, d_rands, temp: _temp, span: _span, alt: _phase);
                // strRet = strRet + _gridProcs.RunAltKernel(d_floats, _span, _phase);
            }
            int[] res = new int[_area];
            //strRet = strRet + _cudaArray.CopyIntsFromDevice(res, d_rands, _area);
            strRet = strRet + _cudaArray.CopyIntsFromDevice(res, d_grid, _area);

            _stopwatch.Stop();

            return new ProcResult<SimGrid<int>>(data:new SimGrid<int>(name:"Update2",
                                                                      width:_span,
                                                                      height:_span,
                                                                      data: res),
                                                err: strRet,
                                                steps:steps,
                                                time: _stopwatch.ElapsedMilliseconds / 1000.0);
        }


        public static ProcResult<SimGrid<int>> Update3(int steps, float temp)
        {
            double t1 = (1.0 / (1.0 + Math.Exp(temp)));
            double t2 = (1.0 / (1.0 + Math.Exp(2*temp)));
            double t3 = (1.0 / (1.0 + Math.Exp(3*temp)));
            double t4 = (1.0 / (1.0 + Math.Exp(4*temp)));

            var strRet = String.Empty;
            _stopwatch.Reset();
            _stopwatch.Start();

            for (var s = 0; s < steps; s++)
            {
                if (_phase == 0)
                {
                    strRet = strRet + _randoProcs.MakeUniformRands(d_rands, _area);
                    _phase = 1;
                }
                else { _phase = 0; }

                strRet = strRet + _gridProcs.RunIsingKernel(
                    d_grid, d_rands, span: _span, alt: _phase, 
                    t1:(float)t1, t2: (float)t2, t3: (float)t3, t4: (float)t4);
               
            }
            int[] res = new int[_area];
            //float[] resf = new float[_area];
            //strRet = strRet + _cudaArray.CopyFloatsFromDevice(resf, d_rands, _area);
            strRet = strRet + _cudaArray.CopyIntsFromDevice(res, d_grid, _area);

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
