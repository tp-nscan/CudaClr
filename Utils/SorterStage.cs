using System;
using System.Collections.Generic;

namespace Utils
{
    public interface ISorterStage : IPermutation
    {
    }

    public class SorterStage : Permutation, ISorterStage
    {
        public SorterStage(int order, IEnumerable<int> terms) : base(order, terms)
        {
        }
    }


    public static class SorterStageEx
    {

        public static ISorterStage RandomSorterStage(this IRando rando, int order)
        {
            var perm = new SorterStage(
                order: order,
                terms: rando.RandomTwoCycle(order)
            );

            return perm;
        }

        public static Tuple<bool, IPermutation> Sort(this ISorterStage stage, IPermutation perm)
        {
            var aRet = new int[stage.Order];

            for (var i = 0; i < stage.Order; i++)
            {
                aRet[i] = perm[i];
            }

            var wasUsed = false;
            for (var i = 0; i < stage.Order; i++)
            {
                var m = stage[i];
                if (m > i)
                {
                    var llv = perm[i];
                    var hlv = perm[m];
                    if (llv > hlv)
                    {
                        aRet[m] = llv;
                        aRet[i] = hlv;
                        wasUsed = true;
                    }
                }
            }
            return new Tuple<bool, IPermutation>(wasUsed, PermutationEx.MakePermutation(aRet));
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
