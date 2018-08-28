﻿using System;
using System.Collections.Generic;
using System.Linq;

namespace Utils
{
    public interface IPermutation
    {
        int Order { get; }
        int this[int index] { get; }
    }

    public static class PermutationEx
    {
        public static Permutation Identity(int order)
        {
            return new Permutation(
                order: order,
                terms: Enumerable.Range(0, order).ToArray());
        }

        public static IPermutation MakePermutation(int[] terms)
        {
            return new Permutation(terms.Length, terms);
        }

        public static Func<IPermutation, object> PermutationIndexComp(int index)
        {
            return p => p[index];
        }

        public static CompositeDictionary<IPermutation, int> MakeCompositeDictionaryForPermutation(int order)
        {
            return new CompositeDictionary<IPermutation, int>(
                    Enumerable.Range(0, order).Select(PermutationIndexComp).ToArray()
                );
        }

        public static bool IsEqualTo(this IPermutation lhs, IPermutation rhs)
        {
            if (lhs.Order != rhs.Order)
            {
                return false;
            }

            for (var i = 0; i < lhs.Order; i++)
            {
                if (lhs[i] != rhs[i])
                {
                    return false;
                }
            }

            return true;
        }

        public static Permutation RandomPermutation(this IRando rando, int order)
        {
            return new Permutation(
                order: order,
                terms: rando.FisherYatesShuffle(Enumerable.Range(0, order).ToArray()));
        }


        public static int OutOfOrderliness(this IPermutation perm)
        {
            var tot = 0;
            for (var i = 0; i < perm.Order; i++)
            {
                tot += (int)Math.Pow((i - perm[i]), 2);
            }
            return tot;
        }


        public static IPermutation Multiply(this IPermutation lhs, IPermutation rhs)
        {
            if (lhs.Order != rhs.Order)
            {
                throw new ArgumentException("The two Permutation must have the same Order");
            }
            var aRet = new int[lhs.Order];

            for (var i = 0; i < lhs.Order; i++)
            {
                aRet[i] = lhs[rhs[i]];
            }
            
            return new Permutation(
                order: lhs.Order,
                terms: aRet
                );
        }

    }

    public class Permutation : IPermutation
    {
        public Permutation(int order, IEnumerable<int> terms)
        {
            _elements = terms.ToArray();
            if (Order != order)
            {
                throw new ArgumentException("Order is not equal to length of terms");
            }
        }

        private readonly int[] _elements;

        public int Order => _elements.Length;

        public int this[int index] => _elements[index];
        
    }

}


//public static T[] FisherYatesShuffle<T>(this IReadOnlyList<T> origList, IRando rando)
//{
//var arrayLength = origList.Count;
//var retArray = origList.ToArray();
//    for (var i = arrayLength - 1; i > 0; i--)
//{
//    var j = rando.NextInt(i + 1);
//    var temp = retArray[i];
//    retArray[i] = retArray[j];
//    retArray[j] = temp;
//}
//return retArray;
//}

//public static T[] FisherYatesPartialShuffle<T>(this IReadOnlyList<T> origList, IRando rando, double mixingRate)
//{
//var arrayLength = origList.Count;
//var retArray = origList.ToArray();
//for (var i = arrayLength - 1; i > 0; i--)
//{
//if (rando.NextDouble() > mixingRate) continue;
//var j = rando.NextInt(i + 1);
//var temp = retArray[i];
//retArray[i] = retArray[j];
//retArray[j] = temp;
//}
//return retArray;
//}
