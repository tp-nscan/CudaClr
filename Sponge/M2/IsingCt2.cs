using System;
using CuArrayClr;
using GridProcsClr;
using RandoClr;

namespace Sponge.M2
{
    public class IsingCt2
    {
        private const int SEED = 123;
        private static IntPtr d_In;
        private static IntPtr d_Out;
        private static IntPtr d_Rands;
        private static bool _backwards;
        private static uint _span;
        private static uint _area;
        private static CudaArray _cudaArray;
        private static GridProcs _gridProcs;
        private static RandoProcs _randoProcs;

        public static string Init(float[] inputs, uint span)
        {
            _span = span;
            _area = _span * _span;
            _backwards = false;

            d_In = new IntPtr();
            d_Out = new IntPtr();
            d_Rands = new IntPtr();

            _cudaArray = new CudaArray();
            _gridProcs = new GridProcs();
            _randoProcs = new RandoProcs();

            var strRet = _cudaArray.ResetDevice();

            // In-Out grids.
            strRet = strRet + _cudaArray.MallocIntsOnDevice(ref d_In, _area);
            strRet = strRet + _cudaArray.CopyFloatsToDevice(inputs, d_In, _area);
            strRet = strRet + _cudaArray.MallocIntsOnDevice(ref d_Out, _area);

            // rando stuff
            strRet = strRet + _randoProcs.MakeGenerator64(SEED);
            strRet = strRet + _cudaArray.MallocFloatsOnDevice(ref d_Rands, _area);

            return strRet;
        }

        public static string Update(float[] results, int steps, float stepSize, float noise)
        {
            var strRet = String.Empty;

            for (int i = 0; i < steps; i++)
            {
                if (_backwards)
                {
                    strRet = _randoProcs.MakeNormalRands(d_Rands, _area, mean: 0.0f, stdev: 1.0f);
                    if (!String.IsNullOrEmpty(strRet))
                    {
                        return strRet;
                    }
                    strRet = _gridProcs.RunCa9fK(d_In, d_Out, d_Rands, _span, stepSize, noise);
                    if (!String.IsNullOrEmpty(strRet))
                    {
                        return strRet;
                    }
                }
                else
                {
                    strRet = _randoProcs.MakeNormalRands(d_Rands, _area, mean:0.0f, stdev:1.0f);
                    if (!String.IsNullOrEmpty(strRet))
                    {
                        return strRet;
                    }

                    strRet = _gridProcs.RunCa9fK(d_Out, d_In, d_Rands, _span, stepSize, noise);
                    if (!String.IsNullOrEmpty(strRet))
                    {
                        return strRet;
                    }
                }

                _backwards = !_backwards;
            }

            if (_backwards)
            {
                strRet = _cudaArray.CopyFloatsFromDevice(results, d_Out, _area);
            }
            else
            {
                strRet = _cudaArray.CopyFloatsFromDevice(results, d_In, _area);
            }

            return strRet;
        }

    }
}
