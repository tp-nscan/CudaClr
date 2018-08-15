using System;

namespace Utils
{
    public static class FloatArrayGen
    {
        public static float DistFromCenter(int index, uint span)
        {
            var cv = span / 2;
            var xv = index % span;
            var yv = (index - xv) / span;
            return (float)Math.Sqrt((xv - cv) * (xv - cv) + (yv - cv) * (yv - cv));
        }

        public static float[] DoubleRing(uint innerD, uint midD, uint outerD, uint span, uint modulus,
                float ringVal, float bkgVal)
        {
            var outputs = new float[span * span];

            for (uint i = 0; i < span; i++)
            {
                for (uint j = 0; j < span; j++)
                {
                    uint index = i * span + j;
                    var dist = IntArrayGen.IntDistFromCenter(index: index, span: span);
                    outputs[index] = ((dist < innerD) || ((dist > midD) && (dist < outerD))) ? ringVal : bkgVal;
                }
            }
            return outputs;
        }

        public static float[] SignedUnitUniformRands(uint span, int seed)
        {
            var outputs = new float[span * span];
            var rando = new Random(seed);
            for (var i = 0; i < span; i++)
            {
                for (var j = 0; j < span; j++)
                {
                    var index = i * span + j;
                    outputs[index] = (float)rando.NextDouble() * 2.0f - 1.0f;
                }
            }
            return outputs;
        }

        public static float[] UnSignedUnitUniformRands(uint span, int seed)
        {
            var outputs = new float[span * span];
            var rando = new Random(seed);
            for (var i = 0; i < span; i++)
            {
                for (var j = 0; j < span; j++)
                {
                    var index = i * span + j;
                    outputs[index] = (float)rando.NextDouble();
                }
            }
            return outputs;
        }

        public static float[] SplitScreen(uint span, float left_val, float right_val)
        {
            var outputs = new float[span * span];
            for (var i = 0; i < span; i++)
            {
                for (var j = 0; j < span/2; j++)
                {
                    var index = i * span + j;
                    outputs[index] = left_val;
                }

                for (var j = span/2; j < span; j++)
                {
                    var index = i * span + j;
                    outputs[index] = right_val;
                }

            }
            return outputs;
        }

        public static float[] LeftRightGradient(uint span, float low_val, float high_val)
        {
            float delta = (high_val - low_val) / (span / 2.0f);
            uint hs = span / 2;

            var outputs = new float[span * span];

            for (var i = 0; i < span; i++)
            {
                for (var j = 0; j < hs; j++)
                {
                    var index = i * span + j;
                    outputs[index] = high_val - j * delta;
                }

                for (var j = hs; j < span; j++)
                {
                    var index = i * span + j;
                    outputs[index] = low_val + (j - hs) * delta;
                }

            }
            return outputs;
        }




    }
}
