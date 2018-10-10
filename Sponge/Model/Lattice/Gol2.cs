using System;
using CuArrayClr;
using GridProcsClr;

namespace Sponge.Model.Lattice
{
    public class Gol2
    {
        private static IntPtr d_In;
        private static IntPtr d_Out;
        private static bool _backwards;
        private static uint _span;
        private static uint _area;
        private static CudaArray _cudaArray;
        private static GridProcs _gridProcs;

        public static string Init(int[] inputs, uint span)
        {
            _span = span;
            _area = _span * _span;
            _backwards = false;

            d_In = new IntPtr();
            d_Out = new IntPtr();

            _cudaArray = new CudaArray();
            _gridProcs = new GridProcs();

            var res = _cudaArray.MallocIntsOnDevice(ref d_In, _area);
            res = res + _cudaArray.CopyIntsToDevice(inputs, d_In, _area);
            res = res + _cudaArray.MallocIntsOnDevice(ref d_Out, _area);

            return res;
        }

        public static string Update(int[] results, int steps)
        {
            var strRet = String.Empty;

            for (int i = 0; i < steps; i++)
            {
                if (_backwards)
                {
                    strRet = _gridProcs.Runk_Gol(d_In, d_Out, _span);
                    if(! String.IsNullOrEmpty(strRet))
                    {
                        return strRet;
                    }
                }
                else
                {
                    strRet = _gridProcs.Runk_Gol(d_Out, d_In, _span);
                    if (! String.IsNullOrEmpty(strRet))
                    {
                        return strRet;
                    }
                }

                _backwards = !_backwards;
            }

            if (_backwards)
            {
                strRet = _cudaArray.CopyIntsFromDevice(results, d_In, _area);
            }
            else
            {
                strRet = _cudaArray.CopyIntsFromDevice(results, d_Out, _area);
            }

            return strRet;
        }

    }
}
