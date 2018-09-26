using System;
using System.Collections.Generic;
using System.Linq;
using Utils.Sortable;

namespace Utils.Sorter
{
    public interface ISorter
    {
        Guid Id { get; }
        Guid GenomeId { get; }
        ISorterStage this[int index] { get; }
        uint Order { get; }
        uint StageCount { get; }
        IEnumerable<ISorterStage> SorterStages { get; }
    }

    public enum StageReplacementMode
    {
        RandomReplace,
        RandomRewire,
        RandomConjugate,
        RCTC
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

        public uint Order => _sorterStages.First().Order;

        public uint StageCount => (uint) _sorterStages.Count();

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
                          .Select(i => randy.ToFullSorterStage(order, i)));
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
                sortable: sortable, 
                result: curPerm);
        }

        public static ISorter MakeSorter(this IEnumerable<ISorterStage> stages, Guid id,
                                         Guid genomeId)
        {
            return new Sorter(
                id: id,
                genomeId: genomeId,
                stages:stages);
        }

        public static ISorter Mutate(this ISorter sorter, IRando rando, StageReplacementMode stageReplacementMode)
        {
            var mutantIndex = rando.NextUint(sorter.StageCount);
            var stageToReplace = sorter[(int) mutantIndex];
            ISorterStage mutantStage = null;

            switch (stageReplacementMode)
            {
                case StageReplacementMode.RandomReplace:
                    mutantStage = rando.ToFullSorterStage(order: sorter.Order, stageNumber: mutantIndex);
                    break;
                case StageReplacementMode.RandomRewire:
                    mutantStage = rando.RewireSorterStage(stageToReplace);
                    break;
                case StageReplacementMode.RandomConjugate:
                    mutantStage = stageToReplace.Conjugate(rando).ToSorterStage(mutantIndex);
                    break;
                case StageReplacementMode.RCTC:
                    mutantStage = stageToReplace.C2c(rando).ToSorterStage(mutantIndex);
                    break;
                default:
                    throw new Exception($"{stageReplacementMode.ToString()}");
            }

            return MakeSorter(
                stages: sorter.SorterStages.ReplaceAtIndex(mutantIndex, mutantStage),
                id: Guid.NewGuid(), 
                genomeId: Guid.Empty);

        }

        public static IEnumerable<ISorter> NextGen(this ISorter sorter,
            IRando rando, int childCount, StageReplacementMode stageReplacementMode, bool cloneOrig)
        {
            var mutantCount = (cloneOrig) ? childCount - 1 : childCount;

            if (cloneOrig)
            {
                yield return sorter;
            }

            for (var i = 0; i < mutantCount; i++)
            {
                yield return sorter.Mutate(rando, stageReplacementMode);
            }
        }

    }
}