using System;
using System.Collections.Generic;
using System.Linq;
using Utils.Sorter;

namespace Utils
{
    public interface IRando
    {   
        /// <summary>
        /// returns a double between 0.0 and 1.0
        /// </summary>
        double NextDouble();
        /// <summary>
        /// returns an int between 0 and < Int.Max
        /// </summary>
        int NextInt();
        int NextInt(int maxVal);
        uint NextUint(uint maxVal);
        uint NextUint();
        bool NextBool(double trueProb);
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

        public static IEnumerable<Tuple<int, bool>> ToIndexedBoolEnumerator(this IRando rando, double trueProbability)
        {
            var curdex = 0;
            while (true)
            {
                yield return new Tuple<int, bool>(curdex++, rando.NextDouble() < trueProbability);
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

        public static uint FixedValue(this uint[] map)
        {
            for (uint i = 0; i < map.Length; i++)
            {
                if (map[i] == i) return i;
            }
            throw new Exception("FixedValue: no fixed value");
        }

        public static void SelectWithoutReplacement<T>(this IRando rando, T[] values, T[] trs)
        {
            if (trs.Length > values.Length)
            {
                throw new Exception("(trs.Length > values.Length) in SelectWithoutReplacement");
            }
            var l = values.ToList();
            while (values.Length - l.Count < trs.Length)
            {
                var np = rando.NextInt(l.Count - 1);
                trs[values.Length - l.Count] = values[np];
                l.RemoveAt(np);
            }
        }

        public static T SelectFromRemaining<T>(this IRando rando, T[] values, bool[] rem)
        {
            while (true)
            {
                var dex = rando.NextInt(values.Length);
                if (rem[dex])
                {
                    rem[dex] = false;
                    return values[dex];
                }
            }
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

        public uint NextUint(uint maxVal)
        {
            return (uint)_random.Next((int)maxVal);
        }

        public uint NextUint()
        {
            _useCount++;
            var rv = (uint) _random.Next();
            var lb = (uint) (_random.Next() % 2);
            return (rv << 1) + lb;
        }

        public bool NextBool(double trueProb)
        {
            _useCount++;
            return (_random.NextDouble() < trueProb);
        }

        public int Seed { get; }

        public int NextInt(int maxVal)
        {
            _useCount++;
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
