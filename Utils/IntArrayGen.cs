using System;
using System.Linq;

namespace Utils
{
    public static class IntArrayGen
    {
        public static int[] ToIntBits(this int[] source)
        {
            return source.Select(i => (i > 0.5) ? 1 : -1).ToArray();
        }


        public static int[] Zippy(int span)
        {
            return Enumerable.Range(0, span * span)
                             .Select(p => (p / span) % 256)
                             .ToArray();
        }


        public static int IntDistFromCenter(int index, int span)
        {
            var cv = span/2;
            var xv = index%span;
            var yv = (index - xv) / span;
            return (int) Math.Sqrt((xv - cv) * (xv - cv) + (yv - cv) * (yv - cv));
        }

        public static int[] BullsEye(int span)
        {
            var outputs = new int[span * span];

            for (var i = 0; i < span; i++)
            {
                for (var j = 0; j < span; j++)
                {
                    var index = i * span + j;
                    outputs[index] = IntDistFromCenter(index: index, span: span) % 256;
                }
            }
            return outputs;
        }


        public static int[] Dot(int span, int modulus)
        {
            var outputs = new int[span * span];

            outputs[span/4 + span*span/2] = modulus/2 + 1;

            return outputs;
        }


        public static int[] Uniform(int len, int value)
        {
            var outputs = new int[len];
            for (var i = 0; i < len; i++)
            {
                outputs[i] = value;
            }
            return outputs;
        }


        public static int[] Spot(int spotSz, int span, int modulus)
        {
            var outputs = new int[span * span];

            for (var i = 0; i < span; i++)
            {
                for (var j = 0; j < span; j++)
                {
                    var index = i * span + j;
                    var dist = IntDistFromCenter(index: index, span: span);
                    //outputs[index] = (dist > spotSz) ? modulus /2 : modulus -1;
                    outputs[index] = (dist > spotSz) ? 0 : modulus / 2 + 1;
                }
            }
            return outputs;
        }


        public static int[] Ring(int innerD, int outerD, int span, int modulus)
        {
            var outputs = new int[span * span];

            for (var i = 0; i < span; i++)
            {
                for (var j = 0; j < span; j++)
                {
                    var index = i * span + j;
                    var dist = IntDistFromCenter(index: index, span: span);
                    //outputs[index] = ((dist > innerD) && (dist < outerD)) ? modulus /2 : modulus -1;
                    outputs[index] = ((dist > innerD) && (dist < outerD)) ? 0 : 1;
                }
            }
            return outputs;
        }

        public static int[] DoubleRing(int innerD, int midD, int outerD, int span, int modulus)
        {
            var outputs = new int[span * span];

            for (var i = 0; i < span; i++)
            {
                for (var j = 0; j < span; j++)
                {
                    var index = i * span + j;
                    var dist = IntDistFromCenter(index: index, span: span);
                    //outputs[index] = ((dist > innerD) && (dist < outerD)) ? modulus /2 : modulus -1;
                    outputs[index] = ((dist < innerD) || ((dist > midD) && (dist < outerD))) ? 3 : modulus / 2;
                }
            }
            return outputs;
        }

        public static int[] MultiRing(int modD, int outerD, int span, int modulus)
        {
            var outputs = new int[span * span];

            for (var i = 0; i < span; i++)
            {
                for (var j = 0; j < span; j++)
                {
                    var index = i * span + j;
                    var dist = IntDistFromCenter(index: index, span: span);
                    //outputs[index] = ((dist > innerD) && (dist < outerD)) ? modulus /2 : modulus -1;
                    outputs[index] = (((dist % modD) < 4) && (dist < outerD)) ? 0 : modulus / 2;
                }
            }
            return outputs;
        }

        public static int[] RandInts(int seed, int span, int min, int max)
        {
            var range = max - min;
            var randy = new Random(seed);
            var outputs = new int[span * span];

            for (var i = 0; i < span * span; i++)
            {
                outputs[i] = randy.Next(range) - min;
            }
            return outputs;
        }


        public static int[] RandInts2(int seed, int arrayLen, double fracOnes)
        {
            var randy = new Random(seed);
            var outputs = new int[arrayLen];

            for (var i = 0; i < arrayLen; i++)
            {
                outputs[i] = (randy.NextDouble() > fracOnes) ? 0 : 1;
            }
            return outputs;
        }

        public static int[] RandInts3(int seed, uint span, uint blockSize, double fracOnes)
        {
            var randy = new Random(seed);
            var outputs = new int[span * span];

            for (var i = 0; i < span; i++)
            {
                var pop = (i / blockSize) %2;
                for (var j = 0; j < span; j++)
                {
                    var dex = i*span + j;
                    var ping = (j / blockSize) % 2;
                    if (((pop + ping)%2) == 1)
                    {
                        outputs[dex] = (randy.NextDouble() > fracOnes) ? 0 : 1;
                    }
                }
            }

            return outputs;
        }

        public static int[] Copy(int[] array, int len)
        {
            var iRet = new int[len];
            for (var i = 0; i < len; i++)
            {
                iRet[i] = array[i];
            }
            return iRet;
        }

        public static void CopyTo(int[] dest, int[] src, int len)
        {
            for (var i = 0; i < len; i++)
            {
                dest[i] = src[i];
            }
        }

    }
}
