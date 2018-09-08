using System;
using System.Collections.Generic;
using System.Linq;

namespace Utils
{
    public interface ISorter
    {
        Guid Id { get; }
        ISorterStage this[int index] { get; }
        int StageCount { get; }
        IEnumerable<ISorterStage> SorterStages { get; }
    }

    public class Sorter : ISorter
    {
        public Sorter(Guid id, IEnumerable<ISorterStage> stages)
        {
            Id = id;

            var curStageNumber = 0;
            _sorterStages = stages.Select(
                    s => new SorterStage(
                    order: s.Order,
                    terms: s.GetMap(),
                    stageNumber: curStageNumber++
                ) as ISorterStage).ToList();
        }
         
        readonly List<ISorterStage> _sorterStages;

        public Guid Id { get; }

        public ISorterStage this[int index] => _sorterStages[index];

        public int StageCount => _sorterStages.Count();

        public IEnumerable<ISorterStage> SorterStages => _sorterStages;
    }

    public class SorterEqualityComparer : IEqualityComparer<ISorter>
    {
        public bool Equals(ISorter lhs, ISorter rhs)
        {
            if (lhs.StageCount != rhs.StageCount)
            {
                return false;
            }

            for (var i = 0; i < lhs.StageCount; i++)
            {
                if (!lhs[i].IsEqualTo(rhs[i]))
                {
                    return false;
                }
            }

            return true;
        }

        int IEqualityComparer<ISorter>.GetHashCode(ISorter obj)
        {
            return GetHashCode(obj);
        }

        public static int GetHashCode(ISorter obj)
        {
            int hCode = 113377;
            int fixo = 733211;
            for (var i = 0; i < obj.StageCount; i++)
            {

                hCode = (hCode * (SorterStageEx.GetHashCode(obj[i]) + fixo)) % hCode;
            }
            return hCode;
        }
    }


    public static class SorterEx
    {

        public static ISorter RandomSorter(this IRando randy, int order, int stageCount)
        {
            return new Sorter(
                id:Guid.NewGuid(), 
                stages: Enumerable.Range(0, stageCount)
                                  .Select(i => randy.RandomFullSorterStage(order, i)));
        }

        public static SortResult Sort(this ISorter sorter, ISortable sortable)
        {
            var stageUse = new bool[sorter.StageCount];
            var curPerm = sortable.GetPermutation();
            for (var i = 0; i < sorter.StageCount; i++)
            {
                var res = sorter[i].Sort(curPerm);
                stageUse[i] = res.Item1;
                curPerm = res.Item2;
            }

            return new SortResult(
                sorter: sorter,
                sortedness: curPerm.Sortedness(), 
                stageUse: stageUse, 
                input: sortable, 
                output:null);
        }


        public static ISorter MakeSorter(this IEnumerable<ISorterStage> stages)
        {
            return new Sorter(
                id: Guid.NewGuid(), 
                stages:stages);
        }

        public static ISorter Copy(this ISorter sorter)
        {
            return MakeSorter(sorter.SorterStages);
        }


        public static ISorter ReplaceStage(this ISorter sorter, ISorterStage sorterStage, int beforeIndex)
        {
            var stages = 
                sorter.SorterStages
                      .Take(beforeIndex)
                      .Concat(sorterStage.AsEnumerable())
                      .Concat(sorter.SorterStages.Skip(beforeIndex + 1));

            return MakeSorter(stages: stages);
        }


        public static ISorter Mutate(this ISorter sorter, IRando rando)
        {
            var mutantIndex = rando.NextInt(sorter.StageCount);
            var mutantStage = rando.MutateSorterStage(sorter[mutantIndex]);
            return ReplaceStage(sorter, mutantStage, mutantIndex);
        }


        public static IEnumerable<ISorter> NextGen(this ISorter sorter, IRando rando, int childCount)
        {
            yield return sorter;
            for (var i = 0; i < childCount - 1; i++)
            {
                yield return sorter.Mutate(rando);
            }
        }


    }
}