using System;
using Sponge.Common;
using Hybridizer.Runtime.CUDAImports;

namespace Sponge.Model
{
    public class IsingProcs
    {
        private static dynamic _wrapper;
        private static IntResidentArray _rands;
        private static IntResidentArray _iaIn;
        private static IntResidentArray _iaOut;
        private const int Primus = 10007;
        private static bool _backwards;
        private static int _span;
        private static int _area;

        [HybridizerIgnore]
        public static void SetupRando()
        {
            _rands = new IntResidentArray(_area);
            var rando = new Random();
            for (var i = 0; i < _area; i++)
            {
                _rands[i] = rando.Next() % Primus;
            }
            _rands.RefreshDevice();
        }

        public static void Init(int[] inputs, int span)
        {
            _span = span;
            _area = _span * _span;
            _backwards = false;

            _iaIn = new IntResidentArray(_area);
            _iaOut = new IntResidentArray(_area);
            for (var i = 0; i < _area; i++)
            {
                _iaIn[i] = inputs[i];
            }
            _iaIn.RefreshDevice();
        }

        public static int[] Update(int steps, double stepSize, double temp)
        {
            var runner = HybRunner.Cuda("Sponge_CUDA.dll").SetDistrib(16, 16, 8, 8, 1, 0);
            _wrapper = runner.Wrap(new IsingProcs());

            for (int i = 0; i < steps; i++)
            {
                if (_backwards)
                {
                    _wrapper.Ic(
                        output: _iaIn,
                        input: _iaOut,
                        span: _span,
                        temp: temp);
                }
                else
                {
                    _wrapper.Ic(
                        output: _iaOut,
                        input: _iaIn,
                        span: _span,
                        temp: temp);
                }

                _backwards = !_backwards;
            }

            return (_backwards) ? HybStuff.CopyIntResidentArray(_iaIn) : HybStuff.CopyIntResidentArray(_iaOut);
        }



        [EntryPoint("Ic")]
        public static void Ic(IntResidentArray output, IntResidentArray input, int span, double temp)
        {
            for (int i = threadIdx.y + blockDim.y * blockIdx.y; i < span; i += gridDim.y * blockDim.y)
            {
                for (int j = threadIdx.x + blockIdx.x * blockDim.x; j < span; j += blockDim.x * gridDim.x)
                {
                    int offset = i * span + j;

                    int im = (i - 1 + span) % span;
                    int ip = (i + 1) % span;
                    int jm = (j - 1 + span) % span;
                    int jp = (j + 1) % span;

                    int topl = input[im * span + jm];
                    int top = input[im * span + j];
                    int topr = input[im * span + jp];
                    int l = input[i * span + jm];
                    int c = input[offset];
                    int r = input[i * span + jp];
                    int botl = input[ip * span + jm];
                    int bot = input[ip * span + j];
                    int botr = input[ip * span + jp];

                    int sum = topl + top + topr + l + r + botl + bot + botr;

                    if (c == 0)
                    {
                        output[offset] = (sum == 3) ? 1 : 0;
                    }
                    else
                    {
                        output[offset] = ((sum == 2) || (sum == 3)) ? 1 : 0;
                    }
                }
            }
        }
    }
}
