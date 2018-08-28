using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Utils
{
    public interface IRando
    {
        double NextDouble();
        int NextInt();
        int NextInt(int maxVal);
        int Seed { get; }
        long UseCount { get; }
    }

    public static class Rando
    {
        public static IRando Standard(int seed)
        {
            return new RandoReg(seed);
        }

        #region int

        public static IEnumerable<int> ToIntEnumerator(this IRando rando, int maxValue)
        {
            while (true)
            {
                yield return rando.NextInt(maxValue);
            }
        }

        public static IEnumerable<int> ToIntEnumerator(this IRando rando)
        {
            while (true)
            {
                yield return rando.NextInt();
            }
        }

        #endregion

        public static IEnumerable<bool> ToBoolEnumerator(this IRando rando, double trueProbability)
        {
            while (true)
            {
                yield return rando.NextDouble() < trueProbability;
            }
        }

        public static IEnumerable<double> ToDoubleEnumerator(this IRando rando)
        {
            while (true)
            {
                yield return rando.NextDouble();
            }
        }

        #region guid

        public static Guid NextGuid(this IRando rando)
        {
            return new Guid
                (
                    (uint)rando.NextInt(),
                    (ushort)rando.NextInt(),
                    (ushort)rando.NextInt(),
                    (byte)rando.NextInt(),
                    (byte)rando.NextInt(),
                    (byte)rando.NextInt(),
                    (byte)rando.NextInt(),
                    (byte)rando.NextInt(),
                    (byte)rando.NextInt(),
                    (byte)rando.NextInt(),
                    (byte)rando.NextInt()
                );
        }


        public static IEnumerable<Guid> ToGuidEnumerator(this IRando rando)
        {
            while (true)
            {
                yield return new Guid
                    (
                        (uint)rando.NextInt(),
                        (ushort)rando.NextInt(),
                        (ushort)rando.NextInt(),
                        (byte)rando.NextInt(),
                        (byte)rando.NextInt(),
                        (byte)rando.NextInt(),
                        (byte)rando.NextInt(),
                        (byte)rando.NextInt(),
                        (byte)rando.NextInt(),
                        (byte)rando.NextInt(),
                        (byte)rando.NextInt()
                    );
            }
        }

        #endregion

        public static IEnumerable<T> Pick<T>(this IRando rando, IReadOnlyList<T> items)
        {
            while (true)
            {
                yield return items[rando.NextInt(items.Count)];
            }
        }

        public static T[] FisherYatesShuffle<T>(this IRando rando, IReadOnlyList<T> items)
        {
            var arrayLength = items.Count;
            var retArray = items.ToArray();
            for (var i = arrayLength - 1; i > 0; i--)
            {
                var j = rando.NextInt(i + 1);
                var temp = retArray[i];
                retArray[i] = retArray[j];
                retArray[j] = temp;
            }
            return retArray;
        }

        static int WalkAndTag(int[] lane, int start, int steps)
        {
            int curSpot = start;
            int remainingSteps = steps;
            while (remainingSteps > 0)
            {
                curSpot++;
                if (lane[curSpot] == -1)
                {
                    remainingSteps--;
                }

                if (curSpot > lane.Length)
                {
                    throw new Exception("curSpot > lane.Length");
                }
            }

            lane[curSpot] = start;

            return curSpot;
        }

        public static int[] RandomTwoCycle(this IRando rando, int order)
        {
            var aRet = Enumerable.Repeat(-1, order).ToArray();
            int curDex = 0;
            int rem = order;
            while (rem > 0)
            {
                if (aRet[curDex] == -1)
                {
                    var steps = rando.NextInt(rem);
                    if (steps == 0)
                    {
                        aRet[curDex] = curDex;
                        rem--;
                    }
                    else
                    {
                        var wr = WalkAndTag(aRet, curDex, steps);
                        aRet[curDex] = wr;
                        rem -= 2;
                    }
                }

                curDex++;
            }
            return aRet;
        }


        public static IEnumerable<double> ExpDist(this IRando rando, double max)
        {
            var logMax = Math.Log(max);
            while (true)
            {
                yield return Math.Exp(rando.NextDouble() * logMax);
            }
        }

        public static IEnumerable<double> PowDist(this IRando rando, double max, double pow)
        {
            while (true)
            {
                yield return Math.Pow(rando.NextDouble(), pow) * max;
            }
        }

    }

    internal class RandoReg : IRando
    {
        private readonly Random _random;

        public RandoReg(int seed)
        {
            Seed = seed;
            _random = new Random(seed);
        }

        public int Seed { get; }

        public int NextInt(int maxVal)
        {
            return _random.Next(maxVal);
        }

        public double NextDouble()
        {
            _useCount++;
            return _random.NextDouble();
        }
        public int NextInt()
        {
            _useCount++;
            return _random.Next();
        }

        private long _useCount;
        public long UseCount => _useCount;
        
    }

}
