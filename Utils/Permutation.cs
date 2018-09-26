using System;
using System.Collections.Generic;
using System.Linq;

namespace Utils
{
    public interface IPermutation
    {
        uint Order { get; }
        uint this[uint index] { get; }
    }

    public class Permutation : IPermutation
    {
        public Permutation(uint order, IEnumerable<uint> terms)
        {
            _elements = terms.ToArray();
            if (Order != order)
            {
                throw new ArgumentException("Order is not equal to length of terms");
            }
        }

        private readonly uint[] _elements;

        public uint Order => (uint) _elements.Length;

        public uint this[uint index] => _elements[index];
        
    }


    public static class PermutationEx
    {
        public static uint[] GetMap(this IPermutation permutation)
        {
            return 0u.CountUp(permutation.Order)
                             .Select(i => permutation[i])
                             .ToArray();
        }

        public static Permutation Identity(uint order)
        {
            return new Permutation( order: order, terms: 0u.CountUp(order));
        }

        public static IPermutation ToInverse(this IPermutation perm)
        {
            var invs = new uint[perm.Order];
            for (uint i = 0; i < perm.Order; i++)
            {
                invs[perm[i]] = i;
            }
            return new Permutation(order:perm.Order, terms: invs);
        }

        public static IPermutation ToConjugate(this IPermutation perm, IPermutation conj)
        {
            return conj.ToInverse().Multiply(perm.Multiply(conj));
        }

        public static IPermutation Conjugate(this IPermutation perm, IRando randy)
        {
            return perm.ToConjugate(randy.ToPermutation(perm.Order));
        }

        public static IPermutation C2c(this IPermutation perm, IRando randy)
        {
            return perm.ToConjugate(randy.ToSingleTwoCyclePermutation(perm.Order));
           // return perm.ToConjugate(randy.ToFullTwoCyclePermutation(perm.Order));
        }

        public static IPermutation ToFullTwoCyclePermutation(this IRando randy, uint order)
        {
            return new Permutation(order, randy.ToFullTwoCycleArray(order));
        }

        public static IPermutation ToSingleTwoCyclePermutation(this IRando randy, uint order)
        {
            return new Permutation(order, randy.ToSingleTwoCycleArray(order));
        }

        public static IPermutation MakePermutation(uint[] terms)
        {
            return new Permutation((uint) terms.Length, terms);
        }

        public static int GetHashCode(this IPermutation perm)
        {
            uint hCode = 0;
            for (uint i = 0; i < perm.Order; i++)
            {
                hCode ^= perm[i];

            }
            return (int)hCode;
        }

        public static IPermutation Mutate(this Permutation permutation, IRando rando, float mutationRate)
        {
            if (rando.NextDouble() < mutationRate)
            {
                return rando.ToPermutation(permutation.Order);
            }
            return permutation;
        }


        public static Func<IPermutation, object> PermutationIndexComp(uint index)
        {
            return p => p[index];
        }

        public static CompositeDictionary<IPermutation, int> PermutationDictionary(uint order)
        {
            return new CompositeDictionary<IPermutation, int>(
                0u.CountUp(order).Select(PermutationIndexComp).ToArray()
            );
        }

        public static bool IsEqualTo(this IPermutation lhs, IPermutation rhs)
        {
            if (lhs.Order != rhs.Order) { return false; }

            for (uint i = 0; i < lhs.Order; i++)
            {
                if (lhs[i] != rhs[i]) { return false; }
            }
            return true;
        }

        public static CompositeDictionary<IPermutation, int> GetOrbit(this IPermutation perm, int maxSize = 1000)
        {
            var pd = PermutationEx.PermutationDictionary(perm.Order);
            var cume = perm;
            for (var i = 0; i < maxSize; i++)
            {
                if (pd.ContainsKey(cume)) { return pd; }

                pd.Add(cume, 1);
                cume = cume.Multiply(perm);
            }

            return pd;
        }

        public static CompositeDictionary<IPermutation, int> ToDistr(this IEnumerable<IPermutation> perms)
        {
            CompositeDictionary<IPermutation, int> pd = null;
            foreach (var perm in perms)
            {
                if (pd == null) pd = PermutationDictionary(perm.Order);
                if (pd.ContainsKey(perm))
                {
                    pd[perm]++;
                }
                else
                {
                    pd.Add(perm, 1);
                }
            }
            return pd;
        }

        public static int OrbitLengthFor(this IPermutation perm, int maxSize = 1000)
        {
            return GetOrbit(perm, maxSize).Count;
        }


        public static IPermutation ToPermutation(this IRando rando, uint order)
        {
            return new Permutation(
                order: order,
                terms: rando.FisherYatesShuffle(0u.CountUp(order).ToArray())
                    .ToArray());
        }


        public static uint Sortedness(this IPermutation perm)
        {
            var tot = 0u;
            for (uint i = 0; i < perm.Order; i++)
            {
                tot += (uint)Math.Pow((i - perm[i]), 2);
            }
            return tot;
        }


        public static IPermutation Multiply(this IPermutation lhs, IPermutation rhs)
        {
            if (lhs.Order != rhs.Order)
            {
                throw new ArgumentException("The two Permutation must have the same Order");
            }
            var aRet = new uint[lhs.Order];

            for (uint i = 0; i < lhs.Order; i++)
            {
                aRet[i] = lhs[rhs[i]];
            }

            return new Permutation(order: lhs.Order, terms: aRet );
        }

        static uint WalkAndTag(uint[] lane, uint start, uint steps)
        {
            var curSpot = start;
            var remainingSteps = steps;
            while (remainingSteps > 0)
            {
                curSpot++;
                if (lane[curSpot] == uint.MaxValue)
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

        public static uint[] ToFullTwoCycleArray(this IRando rando, uint order)
        {
            var aRet = uint.MaxValue.Repeat(order).ToArray();

            var rem = order;
            if (order % 2 == 1)
            {
                var cd = rando.NextUint(rem);
                aRet[cd] = cd;
                rem--;
            }

            var curDex = 0u;
            while (rem > 0)
            {
                if (aRet[curDex] == uint.MaxValue)
                {
                    var steps = rando.NextUint(rem - 1) + 1;
                    var wr = WalkAndTag(aRet, curDex, steps);
                    aRet[curDex] = wr;
                    rem -= 2;
                }
                curDex++;
            }
            return aRet;
        }


        public static uint[] ToSingleTwoCycleArray(this IRando rando, uint order)
        {
            var pair = new uint[3];
            var id = 0u.CountUp(order).ToArray();
            rando.SelectWithoutReplacement(id, pair);
            id[pair[0]] = pair[1];
            id[pair[1]] = pair[2];
            id[pair[2]] = pair[0];

            return id;
        }
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
