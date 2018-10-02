using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using Utils.Sortable;
using Utils.Sorter;

namespace Utils.Ga
{
    public enum SorterGaResultType
    {
        Normal,
        Error
    }

    //public interface ISorterGa
    //{
    //    SorterGaResultType
    //}

    public class SortingPhenotypes
    {
        public SortingPhenotypes(SorterPool sorterPool, SortablePool sortablePool)
        {
            SorterPool = sorterPool;
            SortablePool = sortablePool;
        }

        public SorterPool SorterPool { get; }

        public SortablePool SortablePool { get; }
    }


    public static class GaExt
    {
        public static SortingResults Eval1(this SortingPhenotypes sortingPhenotypes, bool saveSortResults)
        {
            var cb = new ConcurrentBag<SortResult>();

            var sr = sortingPhenotypes.SortablePool.AsParallel()
                .SelectMany(
                sb => sortingPhenotypes.SorterPool.Select(st => new {st, sb }));

            sr.ForAll(t=> cb.Add(t.st.Sort(t.sb)));

            return new SortingResults(cb, saveSortResults);
        }

        public static SortingResults Eval(this SortingPhenotypes sortingPhenotypes, bool saveSortResults)
        {
            var sr = sortingPhenotypes.SortablePool.AsParallel()
                .SelectMany(
                    sb => sortingPhenotypes.SorterPool.Select(st => st.Sort(sb)));

            return new SortingResults(sr, saveSortResults);
        }

        public static SortingResults Evalo(this SortingPhenotypes sortingPhenotypes, bool saveSortResults)
        {
            var sr = sortingPhenotypes.SortablePool.SelectMany(
                sb => sortingPhenotypes.SorterPool.Select(st => st.Sort(sb)));

            return new SortingResults(sr, saveSortResults);
        }

        public static SortingPhenotypes EvolveSorters(this SortingPhenotypes sortingPhenotypes,
            Dictionary<Guid, SorterResult> sorterResults,
            IRando randy,
            int selectionFactor,
            StageReplacementMode stageReplacementMode, bool cloneWinners)
        {
            var winSortersCount = sortingPhenotypes.SorterPool.Count() / selectionFactor;
            var bestSorters = sorterResults.Values
                .OrderBy(r => r.AverageSortedness)
                .Take(winSortersCount)
                .ToList();

            var newSorters = bestSorters.SelectMany(b =>
                b.Sorter.NextGen(randy, selectionFactor, stageReplacementMode, cloneWinners)).ToList();

            return new SortingPhenotypes(
                sorterPool: new SorterPool(Guid.NewGuid(), newSorters),
                sortablePool: sortingPhenotypes.SortablePool);
        }


        public static SortingPhenotypes EvolveSortables(this SortingPhenotypes sortingPhenotypes,
            Dictionary<Guid, SortableResult> sortableResults,
            IRando randy, 
            double replacementRate, 
            bool cloneWinners)
        {
            return (cloneWinners)
                ? sortingPhenotypes.EvolveSortablesAndCloneWinners(sortableResults, randy, replacementRate)
                : sortingPhenotypes.EvolveSortables(sortableResults, randy, replacementRate);
        }


        static SortingPhenotypes EvolveSortablesAndCloneWinners(this SortingPhenotypes sortingPhenotypes,
                        Dictionary<Guid, SortableResult> sortableResults,
                        IRando randy, 
                        double replacementRate)
        {
            var winSortablesCount = (int)(sortingPhenotypes.SortablePool.Count() * (1.0 - replacementRate));
            var looseSortablesCount = sortingPhenotypes.SortablePool.Count() - winSortablesCount;
            var newSortables = Enumerable.Range(0, looseSortablesCount)
                .Select(i => randy.ToPermutation(sortingPhenotypes.SorterPool.Order).ToSortable());

            var bestSortables = sortableResults.Values
                .OrderByDescending(r => r.AverageSortedness)
                .Take(winSortablesCount)
                .Select(sr => sr.Sortable);

            var newSortablePool = new SortablePool(Guid.NewGuid(), bestSortables.Concat(newSortables));

            return new SortingPhenotypes(
                sorterPool: sortingPhenotypes.SorterPool,
                sortablePool: newSortablePool);
        }


        static SortingPhenotypes EvolveSortables(this SortingPhenotypes sortingPhenotypes,
            Dictionary<Guid, SortableResult> sortableResults,
            IRando randy, double replacementRate)
        {
            var winSortablesCount = (int)(sortingPhenotypes.SortablePool.Count() * (1.0 - replacementRate));
            var looseSortablesCount = sortingPhenotypes.SortablePool.Count() - winSortablesCount;
            var newSortables = Enumerable.Range(0, looseSortablesCount)
                .Select(i => randy.ToPermutation(sortingPhenotypes.SortablePool.Order).ToSortable());

            var bestSortables = sortableResults.Values
                .OrderByDescending(r => r.AverageSortedness)
                .Take(winSortablesCount)
                .Select(sr => sr.Sortable);

            var newSortablePool = new SortablePool(Guid.NewGuid(), bestSortables.Concat(newSortables));

            return new SortingPhenotypes(
                sorterPool: sortingPhenotypes.SorterPool,
                sortablePool: newSortablePool);
        }


    }
}