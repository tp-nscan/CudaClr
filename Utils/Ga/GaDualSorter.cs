using System;
using System.Collections.Generic;
using System.Linq;
using Utils.Genome;
using Utils.Sortable;
using Utils.Sorter;

namespace Utils.Ga
{
    public class GaDualSorter
    {
        public GaDualSorter(GenomePoolDualSorter genomePoolDualSorter, 
             SortablePool sortablePool,
             IRando randy)
        {
            SortablePool = sortablePool;
            GenomePoolDualSorter = genomePoolDualSorter;
            SorterPool = new SorterPool(id:Guid.NewGuid(), 
                sorters: GenomePoolDualSorter.SelectMany(
                    gds=> gds.MakePhenotypes(randy, 1)));
        }

        public SorterPool SorterPool { get; }

        public SortablePool SortablePool { get; }

        public GenomePoolDualSorter GenomePoolDualSorter { get; }
    }


    public class GaDualSorterResult
    {
        public GaDualSorterResult(GaDualSorter gaDualSorter, 
            IEnumerable<SortResult> sortResults, bool saveSortResults)
        {
            GaDualSorter = gaDualSorter;

            var srl = sortResults.ToList();

            var gbSorter = srl.GroupBy(sr => sr.Sorter).ToList();
            SorterResults = gbSorter.Select(
                g => new SorterResult(g, saveSortResults)).ToDictionary(g => g.Sorter.Id);

            var gbSortable = srl.GroupBy(sr => sr.Sortable).ToList();
            SortableResults = gbSortable.Select(
                g => new SortableResult(g, saveSortResults)).ToDictionary(g => g.Sortable.Id);

        }

        public GaDualSorter GaDualSorter { get; }

        public Dictionary<Guid, SorterResult> SorterResults { get; }

        public Dictionary<Guid, SortableResult> SortableResults { get; }
    }


    public static class GaDualSorterExt
    {
        public static GaDualSorterResult Eval(this GaDualSorter gaDualSorter, 
            bool saveSortResults)
        {
            var srs = gaDualSorter.SortablePool
                .AsParallel()
                .SelectMany(
                    sb => gaDualSorter.SorterPool.Select(st => st.Sort(sb)));

            return new GaDualSorterResult(
                gaDualSorter: gaDualSorter,
                sortResults:srs, 
                saveSortResults: saveSortResults);
        }

        public static GaDualSorter EvolveSorters(this GaDualSorterResult res, IRando randy, int selectionFactor)
        {
            var winSortersCount = res.GaDualSorter.SorterPool.Count() / selectionFactor;
            var bestSorterResults = res.SorterResults.Values
                                       .OrderBy(r => r.AverageSortedness)
                                       .Take(winSortersCount)
                                       .ToList();
            
            var bestGenomes = bestSorterResults.GroupBy(s => s.Sorter.GenomeId)
                .Select(g => res.GaDualSorter.GenomePoolDualSorter.GenomeDualSorters[g.Key]);

            var newGenomes = bestGenomes.SelectMany(g =>
                Enumerable.Range(0, selectionFactor).Select(i => g.Mutate(randy)));

            return new GaDualSorter(
                genomePoolDualSorter: new GenomePoolDualSorter(Guid.NewGuid(), newGenomes),
                sortablePool: res.GaDualSorter.SortablePool,
                randy: randy);
        }

        public static GaDualSorter EvolveSortersRecomb(this GaDualSorterResult res, IRando randy, int selectionFactor)
        {
            var winSortersCount = res.GaDualSorter.SorterPool.Count() / selectionFactor;
            var bestSorterResults = res.SorterResults.Values
                .OrderBy(r => r.AverageSortedness)
                .Take(winSortersCount)
                .ToList();

            var bestGenomes = bestSorterResults.GroupBy(s => s.Sorter.GenomeId)
                .Select(g => res.GaDualSorter.GenomePoolDualSorter.GenomeDualSorters[g.Key]);

            var bestPairs = bestGenomes.ToRandomPairs(randy);

            var recombies = bestPairs.SelectMany(p => p.Item1.Recombine(p.Item2, randy).Split());

            var newGenomes = recombies.SelectMany(g =>
                Enumerable.Range(0, selectionFactor).Select(i => g.Mutate(randy)));

            return new GaDualSorter(
                genomePoolDualSorter: new GenomePoolDualSorter(Guid.NewGuid(), newGenomes),
                sortablePool: res.GaDualSorter.SortablePool,
                randy: randy);
        }


        public static GaDualSorter EvolveSortables(this GaDualSorterResult res, IRando randy, double replacementRate)
        {
            var winSortablesCount = (int)(res.GaDualSorter.SortablePool.Count() * (1.0 - replacementRate));
            var looseSortablesCount = res.GaDualSorter.SortablePool.Count() - winSortablesCount;
            var newSortables = Enumerable.Range(0, looseSortablesCount)
                .Select(i => randy.ToPermutation(res.GaDualSorter.SorterPool.Order).ToSortable());

            var bestSortables = res.SortableResults.Values
                .OrderByDescending(r => r.AverageSortedness)
                .Take(winSortablesCount)
                .Select(sr => sr.Sortable);

            var newSortablePool = new SortablePool(Guid.NewGuid(), bestSortables.Concat(newSortables));

            return new GaDualSorter(
                genomePoolDualSorter: res.GaDualSorter.GenomePoolDualSorter,
                sortablePool: newSortablePool,
                randy: randy);
        }

    }


}
