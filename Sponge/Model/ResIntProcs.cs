using System;
using Sponge.Common;
using Hybridizer.Runtime.CUDAImports;

namespace Sponge.Model
{
    public class ResIntProcs
    {
        private static dynamic _wrapper;
        private static IntResidentArray _iaInCa;
        private static IntResidentArray _iaOutCa;
        private static IntResidentArray _iaInXy;
        private static IntResidentArray _iaOutXy;
        private const int Primus = 10007;
        private static bool _backwards;
        private static int _span;
        private static int _area;
        private static int _modulus;

        public static void SetupRando()
        {
            _iaInCa = new IntResidentArray(_area);
            var rando = new Random();
            for (var i = 0; i < _area; i++)
            {
                _iaInCa[i] = rando.Next()%Primus;
            }
            _iaInCa.RefreshDevice();
        }

        public static void Init(int[] inputsXy, int[] inputsCa,
            int span, int modulus)
        {
            _span = span;
            _area = _span*_span;
            _modulus = modulus;
            _backwards = false;
            SetupRando();

            _iaInXy = new IntResidentArray(_area);
            _iaOutXy = new IntResidentArray(_area);
            _iaInCa = new IntResidentArray(_area);
            _iaOutCa = new IntResidentArray(_area);

            for (var i = 0; i < _area; i++)
            {
                _iaInXy[i] = inputsXy[i];
                _iaInCa[i] = inputsCa[i];
            }
            _iaInXy.RefreshDevice();
            _iaInCa.RefreshDevice();
        }

        public static int[] Update(int steps, double stepSize, double noise)
        {
            int roughNoise = (int) (noise*_modulus);
            int intNoise = 0;

            if (noise > 0)
            {
                intNoise = roughNoise - (roughNoise%2) + 1;
            }

            var runner = HybRunner.Cuda("Sponge_CUDA.dll").SetDistrib(8, 8, 8, 8, 1, 0);
            _wrapper = runner.Wrap(new ResIntProcs());

            for (int i = 0; i < steps; i++)
            {
                if (_backwards)
                {
                    _wrapper.RiTr(
                        outXy: _iaInXy,
                        inXy: _iaOutXy,
                        outCa: _iaInCa,
                        inCa: _iaOutCa,
                        span: _span,
                        modulus: _modulus,
                        stepSize: stepSize,
                        noise: intNoise);
                }
                else
                {
                    _wrapper.RiTr(
                        outXy: _iaOutXy,
                        inXy: _iaInXy,
                        outCa: _iaOutCa,
                        inCa: _iaInCa,
                        span: _span,
                        modulus: _modulus,
                        stepSize: stepSize,
                        noise: intNoise);
                }

                _backwards = !_backwards;
            }

            return (_backwards) ? HybStuff.CopyIntResidentArray(_iaInXy) : HybStuff.CopyIntResidentArray(_iaOutXy);
           // return (_backwards) ? HybStuff.CopyIntResidentArray(_iaInCa) : HybStuff.CopyIntResidentArray(_iaOutCa);
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


        [EntryPoint("RiTr")]
        public static void RiTr(IntResidentArray outXy, IntResidentArray inXy, IntResidentArray outCa,
            IntResidentArray inCa, int span, int modulus, double stepSize, int noise)
        {
            for (int i = threadIdx.y + blockDim.y*blockIdx.y; i < span; i += gridDim.y*blockDim.y)
            {
                for (int j = threadIdx.x + blockIdx.x*blockDim.x; j < span; j += blockDim.x*gridDim.x)
                {
                    int offset = i*span + j;
                    int im = (i - 1 + span)%span;
                    int ip = (i + 1)%span;
                    int jm = (j - 1 + span)%span;
                    int jp = (j + 1)%span;

                    int top = inXy[im*span + j];
                    int l = inXy[i*span + jm];
                    int c = inXy[offset];
                    int r = inXy[i*span + jp];
                    int bot = inXy[ip*span + j];

                    int resTop = ModTent(c, top, modulus);
                    int resLeft = ModTent(c, l, modulus);
                    int resRight = ModTent(c, r, modulus);
                    int resBottom = ModTent(c, bot, modulus);

                    int xyDelta = resTop + resLeft + resRight + resBottom;


                    top = inCa[im * span + j];
                    l = inCa[i * span + jm];
                    int cC = inCa[offset];
                    r = inCa[i * span + jp];
                    bot = inCa[ip * span + j];

                    int caNext = (top + l + r + bot) % modulus;
                    outCa[offset] = caNext;


                    double diff = ModTent(cC, caNext, modulus);
                    diff = diff/modulus;
                    diff *= noise;

                    diff += xyDelta;
                    diff *= stepSize;
                    outXy[offset] = (c + (int) diff + 4*modulus)%modulus;
                }
            }
        }

        [EntryPoint("RndAcc")]
        public static void RndAcc(IntResidentArray output, IntResidentArray input, IntResidentArray seeds,
            int span, int modulus, double stepSize, int noise)
        {
            for (int i = threadIdx.y + blockDim.y*blockIdx.y; i < span; i += gridDim.y*blockDim.y)
            {
                for (int j = threadIdx.x + blockIdx.x*blockDim.x; j < span; j += blockDim.x*gridDim.x)
                {
                    int offset = i*span + j;
                    int resN = 0;
                    int c = input[offset];

                    int diff = (int) (resN*stepSize);

                    output[offset] = (c + diff + 4*modulus)%modulus;
                }
            }
        }
    }
}
