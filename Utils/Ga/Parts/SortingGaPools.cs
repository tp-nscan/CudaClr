using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using Utils.Sortable;
using Utils.Sorter;

namespace Utils.Ga.Parts
{
    public enum SorterGaResultType
    {
        Normal,
        Error
    }

    public class SortingGaPools
    {
        public SortingGaPools(SorterPool sorterPool, SortablePool sortablePool)
        {
            SorterPool = sorterPool;
            SortablePool = sortablePool;
        }

        public SorterPool SorterPool { get; }

        public SortablePool SortablePool { get; }
    }


    public static class SortingGaPoolsExt
    {
        public static SortingResults Eval1(this SortingGaPools sortingPhenotypes, bool saveSortResults)
        {
            var cb = new ConcurrentBag<SortResult>();

            var sr = sortingPhenotypes.SortablePool.AsParallel()
                .SelectMany(
                sb => sortingPhenotypes.SorterPool.Select(st => new {st, sb }));

            sr.ForAll(t=> cb.Add(t.st.Sort(t.sb)));

            return new SortingResults(cb, saveSortResults);
        }

        public static SortingResults Eval(this SortingGaPools sortingPhenotypes, bool saveSortResults)
        {
            var sr = sortingPhenotypes.SortablePool.AsParallel()
                .SelectMany(
                    sb => sortingPhenotypes.SorterPool.Select(st => st.Sort(sb)));

            return new SortingResults(sr, saveSortResults);
        }

        public static SortingResults Evalo(this SortingGaPools sortingPhenotypes, bool saveSortResults)
        {
            var sr = sortingPhenotypes.SortablePool.SelectMany(
                sb => sortingPhenotypes.SorterPool.Select(st => st.Sort(sb)));

            return new SortingResults(sr, saveSortResults);
        }

        public static SortingGaPools EvolveSorters(this SortingGaPools sortingPhenotypes,
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

            return new SortingGaPools(
                sorterPool: new SorterPool(Guid.NewGuid(), newSorters),
                sortablePool: sortingPhenotypes.SortablePool);
        }


        public static SortingGaPools EvolveSortables(this SortingGaPools sortingPhenotypes,
            Dictionary<Guid, SortableResult> sortableResults,
            IRando randy, 
            double replacementRate, 
            bool cloneWinners)
        {
            return (cloneWinners)
                ? sortingPhenotypes.EvolveSortablesAndCloneWinners(sortableResults, randy, replacementRate)
                : sortingPhenotypes.EvolveSortables(sortableResults, randy, replacementRate);
        }


        static SortingGaPools EvolveSortablesAndCloneWinners(this SortingGaPools sortingPhenotypes,
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

            return new SortingGaPools(
                sorterPool: sortingPhenotypes.SorterPool,
                sortablePool: newSortablePool);
        }


        static SortingGaPools EvolveSortables(this SortingGaPools sortingPhenotypes,
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

            return new SortingGaPools(
                sorterPool: sortingPhenotypes.SorterPool,
                sortablePool: newSortablePool);
        }


    }
}