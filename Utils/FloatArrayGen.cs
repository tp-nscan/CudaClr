using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Utils
{
    public static class FloatArrayGen
    {
        public static float DistFromCenter(int index, int span)
        {
            var cv = span / 2;
            var xv = index % span;
            var yv = (index - xv) / span;
            return (float)Math.Sqrt((xv - cv) * (xv - cv) + (yv - cv) * (yv - cv));
        }

        public static float[] DoubleRing(int innerD, int midD, int outerD, int span, int modulus)
        {
            var outputs = new float[span * span];

            for (var i = 0; i < span; i++)
            {
                for (var j = 0; j < span; j++)
                {
                    var index = i * span + j;
                    var dist = IntArrayGen.IntDistFromCenter(index: index, span: span);
                    outputs[index] = ((dist < innerD) || ((dist > midD) && (dist < outerD))) ? 0.8f : 0.0f;
                }
            }
            return outputs;
        }

        public static float[] SignedUnitUniformRands(int span, int seed)
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

    }
}
