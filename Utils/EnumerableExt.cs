using System.Collections.Generic;

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


    }
}