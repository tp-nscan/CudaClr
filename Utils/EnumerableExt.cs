using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices.ComTypes;

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

        public static IEnumerable<T> AsEnumerable<T>(this T item)
        {
                yield return item;
        }


        public static int KeyOverlap<K, V>(this Dictionary<K, V> dict, Dictionary<K, V> comp)
        {
            return comp.Keys.Sum(k => (dict.ContainsKey(k) ? 1 : 0));
        }

        public static IEnumerable<T> ReplaceAtIndex<T>(this IEnumerable<T> source, int index,
            T replacement)
        {
            var lst = source.ToList();
            lst[index] = replacement;
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
        public static int LowerTriangularIndex(int row, int col)
        {
            var r = row;
            var c = col;
            if (row < col)
            {
                r = col;
                c = row;
            }
            return (r * (r + 1)) / 2 + col;
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
    }
}