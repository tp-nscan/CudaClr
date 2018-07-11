using System;
using Sponge.Common;
using Hybridizer.Runtime.CUDAImports;

namespace Sponge.Model
{
    public class CaProcs
    {
        private static dynamic _wrapper;
        private static IntResidentArray _rands;
        private static IntResidentArray _iaIn;
        private static IntResidentArray _iaOut;
        private const int Primus = 10007;
        private static bool _backwards;
        private static int _span;
        private static int _area;
        private static int _modulus;

        [HybridizerIgnore]
        public static void SetupRando()
        {
            _rands = new IntResidentArray(_area);
            var rando = new Random();
            for (var i = 0; i < _area; i++)
            {
                _rands[i] = rando.Next()%Primus;
            }
            _rands.RefreshDevice();
        }

        [HybridizerIgnore]
        public static void Init(int[] inputs, int span, int modulus)
        {
            _span = span;
            _area = _span*_span;
            _modulus = modulus;
            _backwards = false;
            SetupRando();

            _iaIn = new IntResidentArray(_area);
            _iaOut = new IntResidentArray(_area);
            for (var i = 0; i < _area; i++)
            {
                _iaIn[i] = inputs[i];
            }
            _iaIn.RefreshDevice();
        }

        [HybridizerIgnore]
        public static int[] Update(int steps, double stepSize, double noise)
        {
            int roughNoise = (int) (noise*_modulus);
            int intNoise = 0;

            if (noise > 0)
            {
                intNoise = roughNoise - (roughNoise%2) + 1;
            }

            var runner = HybRunner.Cuda("Sponge_CUDA.dll").SetDistrib(8, 8, 8, 8, 1, 0);
            _wrapper = runner.Wrap(new CaProcs());

            for (int i = 0; i < steps; i++)
            {
                if (_backwards)
                {
                    _wrapper.CaTr(output: _iaIn,
                        input: _iaOut,
                        seeds: _rands,
                        span: _span,
                        modulus: _modulus,
                        stepSize: stepSize,
                        noise: intNoise);
                }
                else
                {
                    _wrapper.CaTr(output: _iaOut,
                        input: _iaIn,
                        seeds: _rands,
                        span: _span,
                        modulus: _modulus,
                        stepSize: stepSize,
                        noise: intNoise);
                }

                _backwards = !_backwards;
            }

            return (_backwards) ? HybStuff.CopyIntResidentArray(_iaIn) : 
                                  HybStuff.CopyIntResidentArray(_iaOut);
        }


        [Kernel]
        public static int NextRand(IntResidentArray seeds, int offset)
        {
            var res = (seeds[offset]*Primus); // % Primus;
            if (res < 0) res = -res;
            seeds[offset] = res;
            return res;
        }


        [Kernel]
        public static int ModDiff(int from, int to, int modulus)
        {
            int diff;
            if (to > from)
            {
                diff = to - from;
                return (diff > modulus/2) ? diff - modulus : diff;
            }
            diff = from - to;
            return (diff > modulus/2) ? modulus - diff : -diff;
        }


        [Kernel]
        public static int ModTent(int from, int to, int modulus)
        {
            int md = ModDiff(from, to, modulus);
            int ms = modulus/4;

            if (md > ms) return modulus/2 - md;
            if (md > -ms) return md;
            return -md - modulus/2;
        }


        [IntrinsicFunction("__syncthreads")]
        private static void SyncThreads()
        {
        }


        [EntryPoint("CaR")]
        public static void CaR(IntResidentArray output, IntResidentArray input, int span, int modulus, double stepSize,
            double noise)
        {
            for (int i = threadIdx.y + blockDim.y*blockIdx.y; i < span; i += gridDim.y*blockDim.y)
            {
                for (int j = threadIdx.x + blockIdx.x*blockDim.x; j < span; j += blockDim.x*gridDim.x)
                {
                    int im = (i - 1 + span)%span;
                    int ip = (i + 1)%span;
                    int jm = (j - 1 + span)%span;
                    int jp = (j + 1)%span;

                    int offset = i*span + j;
                    int topl = input[im*span + jm];
                    int top = input[im*span + j];
                    int topr = input[im*span + jp];
                    int l = input[i*span + jm];
                    int c = input[offset];
                    int r = input[i*span + jp];
                    int botl = input[ip*span + jm];
                    int bot = input[ip*span + j];
                    int botr = input[ip*span + jp];

                    int res1 = ModTent(c, topl, modulus);
                    int res2 = ModTent(c, top, modulus);
                    int res3 = ModTent(c, topr, modulus);
                    int res4 = ModTent(c, l, modulus);
                    int res5 = ModTent(c, r, modulus);
                    int res6 = ModTent(c, botl, modulus);
                    int res7 = ModTent(c, bot, modulus);
                    int res8 = ModTent(c, botr, modulus);

                    output[offset] = (c + res1 + res2 + res3 + res4 + res5 +
                                      res6 + res7 + res8 + 8*modulus)%modulus;
                    //SyncThreads();
                }
            }
        }


        [EntryPoint("CaT")]
        public static void CaT(IntResidentArray output, IntResidentArray input, int span, int modulus, double stepSize,
            int noise)
        {
            for (int i = threadIdx.y + blockDim.y*blockIdx.y; i < span; i += gridDim.y*blockDim.y)
            {
                for (int j = threadIdx.x + blockIdx.x*blockDim.x; j < span; j += blockDim.x*gridDim.x)
                {
                    int im = (i - 1 + span)%span;
                    int ip = (i + 1)%span;
                    int jm = (j - 1 + span)%span;
                    int jp = (j + 1)%span;

                    int offset = i*span + j;
                    int top = input[im*span + j];
                    int l = input[i*span + jm];
                    int c = input[offset];
                    int r = input[i*span + jp];
                    int bot = input[ip*span + j];

                    int res2 = ModTent(c, top, modulus);
                    int res4 = ModTent(c, l, modulus);
                    int res5 = ModTent(c, r, modulus);
                    int res7 = ModTent(c, bot, modulus);

                    double diff = res2 + res4 + res5 + res7;
                    diff *= stepSize;

                    output[offset] = (c + (int) diff + 4*modulus)%modulus;
                }
            }
        }

        [EntryPoint("CaTr")]
        public static void CaTr(IntResidentArray output, IntResidentArray input, IntResidentArray seeds,
            int span, int modulus, double stepSize, int noise)
        {
            for (int i = threadIdx.y + blockDim.y*blockIdx.y; i < span; i += gridDim.y*blockDim.y)
            {
                for (int j = threadIdx.x + blockIdx.x*blockDim.x; j < span; j += blockDim.x*gridDim.x)
                {
                    int offset = i*span + j;
                    int resN = (NextRand(seeds, offset)%noise) - noise/2;

                    int im = (i - 1 + span)%span;
                    int ip = (i + 1)%span;
                    int jm = (j - 1 + span)%span;
                    int jp = (j + 1)%span;

                    int top = input[im*span + j];
                    int l = input[i*span + jm];
                    int c = input[offset];
                    int r = input[i*span + jp];
                    int bot = input[ip*span + j];

                    int resTop = ModTent(c, top, modulus);
                    int resLeft = ModTent(c, l, modulus);
                    int resRight = ModTent(c, r, modulus);
                    int resBottom = ModTent(c, bot, modulus);

                    double diff = resTop + resLeft + resRight + resBottom + resN;
                    diff *= stepSize;

                    output[offset] = (c + (int) diff + 4*modulus)%modulus;
                }
            }
        }
    }
}
