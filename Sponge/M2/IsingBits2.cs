using System;
using CuArrayClr;
using GridProcsClr;
using RandoClr;

namespace Sponge.M2
{
    public class IsingBits2
    {
        private const int SEED = 123;
        private static IntPtr d_floats;
        private static bool _backwards;
        private static uint _span;
        private static uint _area;
        private static CudaArray _cudaArray;
        private static GridProcs _gridProcs;
        private static RandoProcs _randoProcs;

        public static string Init(int[] inputs, uint span)
        {
            _span = span;
            _area = _span * _span;
            _backwards = false;

            d_floats = new IntPtr();

            _cudaArray = new CudaArray();
            _gridProcs = new GridProcs();
            _randoProcs = new RandoProcs();

            var strRet = _cudaArray.ResetDevice();
            strRet = strRet + _randoProcs.MakeGenerator64(SEED);
            strRet = strRet + _cudaArray.MallocFloatsOnDevice(ref d_floats, _area);
            return strRet;
        }

        public static string Update(int[] results, int steps)
        {
            var strRet = String.Empty;
            for (int i = 0; i < steps; i++)
            {
                strRet = _randoProcs.MakeUniformRands(d_floats, _area);
            }

            float[] res = new float[_area];
            strRet = _cudaArray.CopyFloatsFromDevice(res, d_floats, _area);

            for(var i=0; i<_area; i++)
            {
                results[i] = (res[i] > 0.5f) ? 1 : 0;
            }

            return strRet;
        }

    }
}
