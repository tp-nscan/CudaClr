using System;
using Sponge.Common;
using Hybridizer.Runtime.CUDAImports;

namespace Sponge.Model
{
    //10007 is a prime
    public class RandProcs
    {
        private static dynamic _wrapper;
        private static IntResidentArray _seeds;
        private static IntResidentArray _iaIn;
        private static IntResidentArray _iaOut;
        private static bool _backwards;
        private static int _span;
        private static int _area;
        private static int _modulus;
        private const int Primus = 10007;


        [HybridizerIgnore]
        public static void SetupRando()
        {
            _seeds = new IntResidentArray(_area);
            var rando = new Random();
            for (var i = 0; i < _area; i++)
            {
                _seeds[i] = rando.Next() % Primus;
            }
            _seeds.RefreshDevice();
        }

        public static void Init(int[] inputs, int span, int modulus)
        {
            _span = span;
            _area = _span * _span;
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

        public static int[] Update(int steps, double stepSize, double noise)
        {
            int roughNoise = (int)(noise * _modulus);
            int intNoise = 0;

            if (noise > 0)
            {
                intNoise = roughNoise - (roughNoise % 2) + 1;
            }

            var runner = HybRunner.Cuda("Sponge_CUDA.dll").SetDistrib(8, 8, 8, 8, 1, 0);
            _wrapper = runner.Wrap(new RandProcs());

            for (int i = 0; i < steps; i++)
            {
                if (_backwards)
                {
                    _wrapper.RandoDrift(output: _iaOut,
                        input: _iaIn,
                        seeds: _seeds,
                        span: _span,
                        modulus: _modulus,
                        stepSize: stepSize,
                        noise: intNoise);
                }
                else
                {
                    _wrapper.RandoDrift(output: _iaIn,
                        input: _iaOut,
                        seeds: _seeds,
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


        //[Kernel]
        //public static int NextInt(IntResidentArray inCa, int offset, int cap)
        //{
        //    while (true)
        //    {
        //        var res = (inCa[offset]*Primus); // % Primus;
        //        inCa[offset] = res;
        //        if(res < cap)
        //            return res;
        //    }
        //}

        [Kernel]
        public static int NextInt2(IntResidentArray seeds, int offset)
        {
            var res = seeds[offset] * Primus;
            if (res < 0) res = -res;
            seeds[offset] = res;
            return res;
        }


        [EntryPoint("RandoGen")]
        public static void RandoGen(IntResidentArray output, IntResidentArray seeds,
            int span, int modulus, double stepSize, int noise)
        {
            for (int i = threadIdx.y + blockDim.y * blockIdx.y; i < span; i += gridDim.y * blockDim.y)
            {
                for (int j = threadIdx.x + blockIdx.x * blockDim.x; j < span; j += blockDim.x * gridDim.x)
                {
                    int offset = i * span + j;
                    int resN = (NextInt2(seeds, offset) % noise) - noise / 2;

                    output[offset] = resN;
                }
            }
        }


        [EntryPoint("RandoDrift")]
        public static void RandoDrift(IntResidentArray output, IntResidentArray input, IntResidentArray seeds,
            int span, int modulus, double stepSize, int noise)
        {
            for (int i = threadIdx.y + blockDim.y * blockIdx.y; i < span; i += gridDim.y * blockDim.y)
            {
                for (int j = threadIdx.x + blockIdx.x * blockDim.x; j < span; j += blockDim.x * gridDim.x)
                {
                    int offset = i * span + j;
                    //int resN = (NextInt(inCa, offset, modulus * 9) % noise) - noise / 2 - 1;
                    //outXy[offset] = (inXy[offset] + modulus + (int)(resN * stepSize)) % modulus;
                    int resN = (NextInt2(seeds, offset) % noise) - noise / 2 - 1;
                    output[offset] = (input[offset] + modulus + (int)(resN * stepSize)) % modulus;
                }
            }
        }


        //[IntrinsicType("curandStateMRG32k3a_t")]
        //[IntrinsicIncludeCUDA("curand_kernel.h")]
        //[StructLayout(LayoutKind.Sequential)]
        //public unsafe struct curandStateMRG32k3a_t
        //{
        //    public fixed double s1[3];
        //    public fixed double s2[3];
        //    public int boxmuller_flag;
        //    public int boxmuller_flag_double;
        //    public float boxmuller_extra;
        //    public double boxmuller_extra_double;
        //    [IntrinsicFunction("curand_init")]
        //    public static void curand_init(ulong seed,
        //        ulong subsequence, ulong offset,
        //        out curandStateMRG32k3a_t state)
        //    { throw new NotImplementedException(); }
        //    [IntrinsicFunction("curand")]
        //    public uint curand()
        //    { throw new NotImplementedException(); }
        //    [IntrinsicFunction("curand_log_normal")]
        //    public float curand_log_normal(float mean, float stdev)
        //    { throw new NotImplementedException(); }
        //}

        //public static void Tickle()
        //{
        //    HybRunner runner = HybRunner.Cuda("Sponge_CUDA.dll").SetDistrib(1, 1);
        //    dynamic wrapped = runner.Wrap(new RandProcs());
        //    cuda.DeviceSynchronize();

        //    curandStateMRG32k3a_t curry;
        //    //curandStateMRG32k3a_t.curand_init(123, 21, 123, out curry);
        //    wrapped.curand_init(123, 21, 123, out curry);
        //}


    }

}
