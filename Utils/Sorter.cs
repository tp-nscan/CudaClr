using System;
using System.Collections.Generic;
using System.Linq;

namespace Utils
{
    public interface ISorter
    {
        Guid Id { get; }
        Guid GenomeId { get; }
        ISorterStage this[int index] { get; }
        int StageCount { get; }
        IEnumerable<ISorterStage> SorterStages { get; }
    }

    public class Sorter : ISorter
    {
        public Sorter(Guid id, Guid genomeId, IEnumerable<ISorterStage> stages)
        {
            Id = id;
            GenomeId = genomeId;

            var curStageNumber = 0u;
            _sorterStages = stages.Select(
                    s => new SorterStage(
                    order: s.Order,
                    terms: s.GetMap(),
                    stageNumber: curStageNumber++
                ) as ISorterStage).ToList();
        }
         
        readonly List<ISorterStage> _sorterStages;

        public Guid Id { get; }

        public Guid GenomeId { get; }

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
            for (var i = 0; i < obj.StageCount; i++)
            {
                hCode = 31 * hCode + SorterStageEx.GetHashCode(obj[i]);
            }
            return hCode;
        }
    }


    public static class SorterEx
    {

        public static ISorter ToSorter(this IRando randy, uint order, uint stageCount)
        {
            return new Sorter(
                id:Guid.NewGuid(),
                genomeId: Guid.Empty, 
                stages: 0u.CountUp(stageCount)
                          .Select(i => randy.RandomFullSorterStage(order, i)));
        }

        public static bool IsEqualTo(this ISorter lhs, ISorter rhs)
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

        public static ISorter MakeSorter(this IEnumerable<ISorterStage> stages, Guid id,
                                         Guid genomeId)
        {
            return new Sorter(
                id: id,
                genomeId: genomeId,
                stages:stages);
        }


        public static ISorter Mutate(this ISorter sorter, IRando rando)
        {
            var mutantIndex = rando.NextInt(sorter.StageCount);
            var mutantStage = rando.MutateSorterStage(sorter[mutantIndex]);
            return MakeSorter(
                stages: sorter.SorterStages.ReplaceAtIndex(mutantIndex, mutantStage), 
                id: Guid.NewGuid(), genomeId:Guid.Empty);
        }


        public static IEnumerable<ISorter> NextGen(this ISorter sorter, 
            IRando rando, int childCount)
        {
            yield return sorter;
            for (var i = 0; i < childCount - 1; i++)
            {
                yield return sorter.Mutate(rando);
            }
        }


    }
}