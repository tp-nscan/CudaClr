using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using Utils.Sortable;
using Utils.Sorter;

namespace Utils.Ga
{
    public class Ga
    {
        public Ga(SorterPool sorterPool, SortablePool sortablePool)
        {
            SorterPool = sorterPool;
            SortablePool = sortablePool;
        }

        public SorterPool SorterPool { get; }

        public SortablePool SortablePool { get; }
    }

    public static class GaExt
    {
        public static GaSortingResults Eval1(this Ga ga, bool saveSortResults)
        {
            var cb = new ConcurrentBag<SortResult>();

            var sr = ga.SortablePool.AsParallel()
                .SelectMany(
                sb => ga.SorterPool.Select(st => new {st, sb }));

            sr.ForAll(t=> cb.Add(t.st.Sort(t.sb)));

            return new GaSortingResults(cb, saveSortResults);
        }

        public static GaSortingResults Eval(this Ga ga, bool saveSortResults)
        {
            var sr = ga.SortablePool.AsParallel()
                .SelectMany(
                    sb => ga.SorterPool.Select(st => st.Sort(sb)));

            return new GaSortingResults(sr, saveSortResults);
        }

        public static GaSortingResults Evalo(this Ga ga, bool saveSortResults)
        {
            var sr = ga.SortablePool.SelectMany(
                sb => ga.SorterPool.Select(st => st.Sort(sb)));

            return new GaSortingResults(sr, saveSortResults);
        }

        public static Ga EvolveSorters(this Ga ga,
            Dictionary<Guid, SorterResult> sorterResults,
            IRando randy,
            int selectionFactor,
            StageReplacementMode stageReplacementMode, bool cloneWinners)
        {
            var winSortersCount = ga.SorterPool.Count() / selectionFactor;
            var bestSorters = sorterResults.Values
                .OrderBy(r => r.AverageSortedness)
                .Take(winSortersCount)
                .ToList();

            var newSorters = bestSorters.SelectMany(b =>
                b.Sorter.NextGen(randy, selectionFactor, stageReplacementMode, cloneWinners)).ToList();

            return new Ga(
                sorterPool: new SorterPool(Guid.NewGuid(), newSorters),
                sortablePool: ga.SortablePool);
        }


        public static Ga EvolveSortables(this Ga ga,
            Dictionary<Guid, SortableResult> sortableResults,
            IRando randy, 
            double replacementRate, 
            bool cloneWinners)
        {
            return (cloneWinners)
                ? ga.EvolveSortablesAndCloneWinners(sortableResults, randy, replacementRate)
                : ga.EvolveSortables(sortableResults, randy, replacementRate);
        }


        static Ga EvolveSortablesAndCloneWinners(this Ga ga,
            Dictionary<Guid, SortableResult> sortableResults,
            IRando randy, 
            double replacementRate)
        {
            var winSortablesCount = (int)(ga.SortablePool.Count() * (1.0 - replacementRate));
            var looseSortablesCount = ga.SortablePool.Count() - winSortablesCount;
            var newSortables = Enumerable.Range(0, looseSortablesCount)
                .Select(i => randy.ToPermutation(ga.SorterPool.Order).ToSortable());

            var bestSortables = sortableResults.Values
                .OrderByDescending(r => r.AverageSortedness)
                .Take(winSortablesCount)
                .Select(sr => sr.Sortable);

            var newSortablePool = new SortablePool(Guid.NewGuid(), bestSortables.Concat(newSortables));

            return new Ga(
                sorterPool: ga.SorterPool,
                sortablePool: newSortablePool);
        }


        static Ga EvolveSortables(this Ga ga,
            Dictionary<Guid, SortableResult> sortableResults,
            IRando randy, double replacementRate)
        {
            var winSortablesCount = (int)(ga.SortablePool.Count() * (1.0 - replacementRate));
            var looseSortablesCount = ga.SortablePool.Count() - winSortablesCount;
            var newSortables = Enumerable.Range(0, looseSortablesCount)
                .Select(i => randy.ToPermutation(ga.SorterPool.Order).ToSortable());

            var bestSortables = sortableResults.Values
                .OrderByDescending(r => r.AverageSortedness)
                .Take(winSortablesCount)
                .Select(sr => sr.Sortable);

            var newSortablePool = new SortablePool(Guid.NewGuid(), bestSortables.Concat(newSortables));

            return new Ga(
                sorterPool: ga.SorterPool,
                sortablePool: newSortablePool);
        }


    }
}