using System;
using System.Collections.Generic;
using System.Linq;

namespace Utils
{

    public interface ISorterStage : IPermutation
    {
        int StageNumber { get; }
    }


    public class SorterStage : Permutation, ISorterStage
    {
        public SorterStage(int order, IEnumerable<int> terms, int stageNumber) : base(order, terms)
        {
            StageNumber = stageNumber;
        }

        public int StageNumber { get; }
    }


    public static class SorterStageEx
    {

        public static int GetHashCode(this ISorterStage sorterStage)
        {
            return PermutationEx.GetHashCode(sorterStage);
        }

        public static ISorterStage RandomFullSorterStage(this IRando rando, int order, int stageNumber)
        {
            var perm = new SorterStage(
                order: order,
                terms: rando.RandomFullTwoCycle(order),
                stageNumber: stageNumber
            );

            return perm;
        }

        public static ISorterStage MutateSorterStage(this IRando rando, ISorterStage sorterStage)
        {
            if (sorterStage.Order % 2 == 1)
            {
                return rando.MutateSorterStageOdd(sorterStage);
            }

            return rando.MutateSorterStageEven(sorterStage);
        }

        static ISorterStage MutateSorterStageOdd(this IRando rando, ISorterStage sorterStage)
        {
            var map = sorterStage.GetMap();
            var fv = map.FixedValue();
            var rems = map.Select(i => true).ToArray();
            rems[fv] = false;
            int lv = rando.SelectFromRemaining(map, rems);
            int hv = sorterStage[lv];

            map[lv] = fv;
            map[fv] = lv;
            map[hv] = hv;

            return new SorterStage(
                order:sorterStage.Order, 
                terms:map, 
                stageNumber: sorterStage.StageNumber);
        }

        static ISorterStage MutateSorterStageEven(this IRando rando, ISorterStage sorterStage)
        {
            var map = sorterStage.GetMap();
            var rems = map.Select(i => true).ToArray();
            var aX = rando.SelectFromRemaining(map, rems);
            var aY = map[aX];
            rems[aY] = false;

            var bX = rando.SelectFromRemaining(map, rems);
            var bY = map[bX];
            rems[bY] = false;

            map[aX] = bY;
            map[aY] = bX;
            map[bX] = aY;
            map[bY] = aX;

            return new SorterStage(
                order: sorterStage.Order,
                terms: map, 
                stageNumber: sorterStage.StageNumber);
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
