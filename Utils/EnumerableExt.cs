using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;

namespace Utils
{
    public static class EnumerableExt
    {
        public static IEnumerable<T> Merge<T>(this IEnumerable<T> first, IEnumerable<T> second)
        {
            foreach (var item in first)
                yield return item;

            foreach (var item in second)
                yield return item;
        }

        public static IEnumerable<T> ToRoundRobin<T>(this IEnumerable<T> source)
        {
            var l = source.ToList();
            while (true)
            {
                for (int i = 0; i < l.Count; i++)
                {
                    yield return l[i];
                }
            }
        }

        public static IEnumerable<T> AsEnumerable<T>(this T item)
        {
                yield return item;
        }

        public static int KeyOverlap<K, V>(this Dictionary<K, V> dict, Dictionary<K, V> comp)
        {
            return comp.Keys.Sum(k => (dict.ContainsKey(k) ? 1 : 0));
        }

        public static Dictionary<K,V> Copy<K,V>(this Dictionary<K, V> dictionary)
        {
            var dRet = new Dictionary<K, V>();
            foreach (var k in dictionary.Keys)
            {
                dRet[k] = dictionary[k];
            }
            return dRet;
        }

        public static IEnumerable<T> ReplaceAtIndex<T>(this IEnumerable<T> source, uint index,
            T replacement)
        {
            var lst = source.ToList();
            lst[(int) index] = replacement;
            return lst;
        }

        public static IEnumerable<Tuple<T, T>> ToRandomPairs<T>(this IEnumerable<T> source, IRando randy)
        {
            var srcList = source.ToList();
            while (srcList.Count > 1)
            {
                var dex = randy.NextInt(srcList.Count);
                var p1 = srcList[dex];
                srcList.RemoveAt(dex); dex = randy.NextInt(srcList.Count);
                var p2 = srcList[dex];
                srcList.RemoveAt(dex);
                yield return new Tuple<T, T>(p1, p2);
            }
        }

        public static Tuple<List<T>, List<T>> Recombo<T>(this IRando randy, List<T> lhs, List<T> rhs)
        {
            var bp = randy.NextInt(lhs.Count);
            var rem = lhs.Count - bp;
            return new Tuple<List<T>, List<T>>(
                    item1: lhs.Take(bp).Concat(rhs.Skip(bp).Take(rem)).ToList(),
                    item2: rhs.Take(bp).Concat(lhs.Skip(bp).Take(rem)).ToList()
                );
        }

        public static IEnumerable<T> Split<T>(this Tuple<T, T> pair)
        {
            yield return pair.Item1;
            yield return pair.Item2;
        }

        //For a symmetric matrix represented as lower triangular in row major order.
        public static uint LowerTriangularIndex(uint row, uint col)
        {
            var r = row;
            var c = col;
            if (row < col)
            {
                r = col;
                c = row;
            }
            return (r * (r + 1)) / 2 + c;
        }

        //For a symmetric matrix represented as lower triangular in row major order.
        public static Tuple<int, int> FromLowerTriangularIndex(int index)
        {
            var lb = (int)Math.Sqrt(index*2 - 1);
            var lf = lb * (lb + 1) / 2 - 1;
            if (lf == index)
            {
                return new Tuple<int, int>(lb - 1, lb - 1);
            }

            return new Tuple<int, int>(lb, index - lf - 1);
        }

        //Lower triangular matrix coords in row major order.
        public static IEnumerable<Tuple<uint, uint>> SquareArrayCoords(this uint span)
        {
            return 0u.CountUp(span).SelectMany(r => 0u.CountUp(span)
                .Select(c => new Tuple<uint, uint>(r, c)));
        }

        //Lower triangular matrix coords in row major order.
        public static IEnumerable<Tuple<uint, uint>> LowerTriangularCoords(this uint span)
        {
            return 0u.CountUp(span).SelectMany(r=> r.CountUp(span)
                                  .Select(c=> new Tuple<uint, uint>(r, c)));
        }

        // Counts from [start-1 ... min], descending
        public static IEnumerable<uint> CountDown(this uint start, uint min)
        {
            for (var i = start-1; i >= min; i--) yield return i;
        }

        // Counts from [start ... max], ascending
        public static IEnumerable<uint> CountUp(this uint start, uint max)
        {
            for (var i = start; i < max; i++) yield return i;
        }

        public static IEnumerable<uint> Repeat(this uint item, uint count)
        {
            for (var i = 0; i < count; i++) yield return item;
        }

        public static IEnumerable<T> Recurse<T>(this Func<T, T> func, T seed, int power)
        {
            var retVal = seed;
            yield return retVal;
            for (var i = 0; i < power -1; i++)
            {
                retVal = func(retVal);
                yield return retVal;
            }
        }
    }
}