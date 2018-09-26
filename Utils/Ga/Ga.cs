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


    public class GaResult
    {
        public GaResult(Ga ga, IEnumerable<SortResult> sortResults, bool saveSortResults)
        {
            Ga = ga;

            var srl = sortResults.ToList();

            var gbSorter = srl.GroupBy(sr => sr.Sorter).ToList();
            SorterResults = gbSorter.Select(
                g => new SorterResult(g, saveSortResults)).ToDictionary(g=>g.Sorter.Id);

            var gbSortable = srl.GroupBy(sr => sr.Input).ToList();
            SortableResults = gbSortable.Select(
                g => new SortableResult(g, saveSortResults)).ToDictionary(g => g.Sortable.Id);

        }

        public Ga Ga { get; }

        public Dictionary<Guid, SorterResult> SorterResults { get; }

        public Dictionary<Guid, SortableResult> SortableResults { get; }
    }


    public static class GaExt
    {
        public static GaResult Eval1(this Ga ga, bool saveSortResults)
        {
            var cb = new ConcurrentBag<SortResult>();

            var sr = ga.SortablePool.AsParallel()
                .SelectMany(
                sb => ga.SorterPool.Select(st => new {st, sb }));


            sr.ForAll(t=> cb.Add(t.st.Sort(t.sb)));

            return new GaResult(ga, cb, saveSortResults);
        }

        public static GaResult Eval(this Ga ga, bool saveSortResults)
        {
            var sr = ga.SortablePool.AsParallel()
                .SelectMany(
                    sb => ga.SorterPool.Select(st => st.Sort(sb)));

            return new GaResult(ga, sr, saveSortResults);
        }

        public static GaResult Evalo(this Ga ga, bool saveSortResults)
        {
            var sr = ga.SortablePool.SelectMany(
                sb => ga.SorterPool.Select(st => st.Sort(sb)));

            return new GaResult(ga, sr, saveSortResults);
        }

        public static Ga EvolveSorters(this GaResult res, IRando randy, int selectionFactor,
            StageReplacementMode stageReplacementMode, bool cloneWinners)
        {
            var winSortersCount = res.Ga.SorterPool.Count() / selectionFactor;
            var bestSorters = res.SorterResults.Values
                .OrderBy(r => r.AverageSortedness)
                .Take(winSortersCount)
                .ToList();

            var newSorters = bestSorters.SelectMany(b =>
                b.Sorter.NextGen(randy, selectionFactor, stageReplacementMode, cloneWinners)).ToList();

            return new Ga(
                sorterPool: new SorterPool(Guid.NewGuid(), newSorters),
                sortablePool: res.Ga.SortablePool);
        }


        public static Ga EvolveSortables(this GaResult res, IRando randy, double replacementRate, bool cloneWinners)
        {
            return (cloneWinners)
                ? EvolveSortablesAndCloneWinners(res, randy, replacementRate)
                : EvolveSortables(res, randy, replacementRate);
        }


        static Ga EvolveSortablesAndCloneWinners(this GaResult res, IRando randy, double replacementRate)
        {
            var winSortablesCount = (int)(res.Ga.SortablePool.Count() * (1.0 - replacementRate));
            var looseSortablesCount = res.Ga.SortablePool.Count() - winSortablesCount;
            var newSortables = Enumerable.Range(0, looseSortablesCount)
                .Select(i => randy.ToPermutation(res.Ga.SorterPool.Order).ToSortable());

            var bestSortables = res.SortableResults.Values
                .OrderByDescending(r => r.AverageSortedness)
                .Take(winSortablesCount)
                .Select(sr => sr.Sortable);

            var newSortablePool = new SortablePool(Guid.NewGuid(), bestSortables.Concat(newSortables));

            return new Ga(
                sorterPool: res.Ga.SorterPool,
                sortablePool: newSortablePool);
        }


        static Ga EvolveSortables(this GaResult res, IRando randy, double replacementRate)
        {
            var winSortablesCount = (int)(res.Ga.SortablePool.Count() * (1.0 - replacementRate));
            var looseSortablesCount = res.Ga.SortablePool.Count() - winSortablesCount;
            var newSortables = Enumerable.Range(0, looseSortablesCount)
                .Select(i => randy.ToPermutation(res.Ga.SorterPool.Order).ToSortable());

            var bestSortables = res.SortableResults.Values
                .OrderByDescending(r => r.AverageSortedness)
                .Take(winSortablesCount)
                .Select(sr => sr.Sortable);

            var newSortablePool = new SortablePool(Guid.NewGuid(), bestSortables.Concat(newSortables));

            return new Ga(
                sorterPool: res.Ga.SorterPool,
                sortablePool: newSortablePool);
        }


    }
}