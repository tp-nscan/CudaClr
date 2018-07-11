using Sponge.Common;
using Hybridizer.Runtime.CUDAImports;

namespace Sponge.Model
{
    public class PoincareProcs
    {
        private static dynamic _wrapper;
        private static IntResidentArray _iaIn;
        private static IntResidentArray _iaOut;
        private static bool _backwards;
        private static int _span;
        private static int _area;
        private static int _modulus;

        public static void Init(int[] inputs, int span, int modulus)
        {
            _span = span;
            _area = _span * _span;
            _modulus = modulus;
            _backwards = false;

            _iaIn = new IntResidentArray(_area);
            _iaOut = new IntResidentArray(_area);
            for (var i = 0; i < _area; i++)
            {
                _iaIn[i] = inputs[i];
            }
            _iaIn.RefreshDevice();
        }

        public static int[] Update(int steps)
        {
 
            var runner = HybRunner.Cuda("Sponge_CUDA.dll").SetDistrib(8, 8, 8, 8, 1, 0);
            _wrapper = runner.Wrap(new PoincareProcs());

            for (int i = 0; i < steps; i++)
            {
                if (_backwards)
                {
                    _wrapper.Cross(
                        output: _iaIn,
                        input: _iaOut,
                        span: _span,
                        modulus: _modulus);
                }
                else
                {
                    _wrapper.Cross(
                        output: _iaOut,
                        input: _iaIn,
                        span: _span,
                        modulus: _modulus);
                }

                _backwards = !_backwards;
            }

            return (_backwards) ? HybStuff.CopyIntResidentArray(_iaIn) :
                                  HybStuff.CopyIntResidentArray(_iaOut);
        }

        //public static int[] OneStep(int[] inputs, int span, int modulus)
        //{
        //    IntResidentArray iaIn = new IntResidentArray(span * span);
        //    IntResidentArray iaOut = new IntResidentArray(span * span);
        //    for (int i = 0; i < span * span; i++)
        //    {
        //        iaIn[i] = inputs[i];
        //    }

        //    var runner = HybRunner.Cuda("Sponge_CUDA.dll").SetDistrib(16, 16, 16, 16, 1, 0);
        //    _wrapper = runner.Wrap(new PoincareProcs());

        //    _wrapper.Cross(iaOut, iaIn, span, modulus);

        //    iaOut.RefreshHost();

        //    var outputs = new int[span * span];
        //    for (int i = 0; i < span * span; i++)
        //    {
        //        outputs[i] = iaOut[i];
        //    }
        //    return outputs;
        //}


        //public static int[] EvenSteps(int[] inputs, int span, int steps, int modulus)
        //{
        //    IntResidentArray iaIn = new IntResidentArray(span * span);
        //    IntResidentArray iaOut = new IntResidentArray(span * span);
        //    for (int i = 0; i < span * span; i++)
        //    {
        //        iaIn[i] = inputs[i];
        //    }

        //    var runner = HybRunner.Cuda("Sponge_CUDA.dll").SetDistrib(16, 16, 16, 16, 1, 0);
        //    _wrapper = runner.Wrap(new PoincareProcs());

        //    for (int i = 0; i < steps/2; i++)
        //    {
        //        _wrapper.Cross(iaOut, iaIn, span, modulus);
        //        _wrapper.Cross(iaIn, iaOut, span, modulus);
        //    }

        //    iaIn.RefreshHost();

        //    var outputs = new int[span * span];
        //    for (int i = 0; i < span * span; i++)
        //    {
        //        outputs[i] = iaIn[i];
        //    }
        //    return outputs;
        //}


        [EntryPoint("Ring")]
        public static void Ring(IntResidentArray output, IntResidentArray input, int span, int modulus)
        {
            for (int i = threadIdx.y + blockDim.y * blockIdx.y; i < span; i += gridDim.y * blockDim.y)
            {
                for (int j = threadIdx.x + blockIdx.x * blockDim.x; j < span; j += blockDim.x * gridDim.x)
                {
                    int im = (i - 1 + span) % span;
                    int ip = (i + 1) % span;
                    int jm = (j - 1 + span) % span;
                    int jp = (j + 1) % span;

                    int offset = i * span + j;
                    int topl = input[im * span + jm];
                    int top = input[im * span + j];
                    int topr = input[im * span + jp];
                    int l = input[i * span + jm];
                    int c = input[offset];
                    int r = input[i * span + jp];
                    int botl = input[ip * span + jm];
                    int bot = input[ip * span + j];
                    int botr = input[ip * span + jp];

                    int nbrs = topl + top + topr + l + r + botl + bot + botr;
                    output[offset] = nbrs % modulus;
                }
            }
        }


        [EntryPoint("Cross")]
        public static void Cross(IntResidentArray output, IntResidentArray input, int span, int modulus)
        {
            for (int i = threadIdx.y + blockDim.y * blockIdx.y; i < span; i += gridDim.y * blockDim.y)
            {
                for (int j = threadIdx.x + blockIdx.x * blockDim.x; j < span; j += blockDim.x * gridDim.x)
                {
                    int im = (i - 1 + span) % span;
                    int ip = (i + 1) % span;
                    int jm = (j - 1 + span) % span;
                    int jp = (j + 1) % span;

                    int offset = i * span + j;
                    int top = input[im * span + j];
                    int l = input[i * span + jm];
                    int c = input[offset];
                    int r = input[i * span + jp];
                    int bot = input[ip * span + j];

                    int nbrs = top + l + r + bot;
                    output[offset] = nbrs % modulus;
                }
            }
        }


        [EntryPoint("DblRing")]
        public static void DblRing(IntResidentArray output, IntResidentArray input, int span, int modulus)
        {
            for (int i = threadIdx.y + blockDim.y * blockIdx.y; i < span; i += gridDim.y * blockDim.y)
            {
                for (int j = threadIdx.x + blockIdx.x * blockDim.x; j < span; j += blockDim.x * gridDim.x)
                {
                    int im = (i - 1 + span) % span;
                    int ip = (i + 1) % span;
                    int jm = (j - 1 + span) % span;
                    int jp = (j + 1) % span;

                    int imm = (i - 2 + span) % span;
                    int ipp = (i + 2) % span;
                    int jmm = (j - 2 + span) % span;
                    int jpp = (j + 2) % span;

                    int offset = i * span + j;

                    int ttopll = input[imm * span + jmm];
                    int ttopl = input[imm * span + jm];
                    int ttop = input[imm * span + j];
                    int ttopr = input[imm * span + jp];
                    int ttoprr = input[imm * span + jpp];

                    int topll = input[im * span + jmm];
                    int topl = input[im * span + jm];
                    int top = input[im * span + j];
                    int topr = input[im * span + jp];
                    int toprr = input[im * span + jpp];

                    int ll = input[i * span + jmm];
                    int l = input[i * span + jm];
                    int c = input[offset];
                    int r = input[i * span + jp];
                    int rr = input[i * span + jpp];

                    int botll = input[ip * span + jmm];
                    int botl = input[ip * span + jm];
                    int bot = input[ip * span + j];
                    int botr = input[ip * span + jp];
                    int botrr = input[ip * span + jpp];

                    int bbotll = input[ipp * span + jmm];
                    int bbotl = input[ipp * span + jm];
                    int bbot = input[ipp * span + j];
                    int bbotr = input[ipp * span + jp];
                    int bbotrr = input[ipp * span + jpp];


                    int nbrs = ttopll + ttopl + ttop + ttopr + ttoprr +
                               topll + topl + top + topr + toprr +
                               ll + l + r + rr +
                               botll + botl + bot + botr + botrr +
                               bbotll + bbotl + bbot + bbotr + bbotrr;

                    output[offset] = nbrs % modulus;
                }
            }
        }
    }
}
