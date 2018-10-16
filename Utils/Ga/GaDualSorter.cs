using System;
using System.Collections.Generic;
using System.Linq;
using Utils.Genome;
using Utils.Genome.Sorter;
using Utils.Sortable;
using Utils.Sorter;

namespace Utils.Ga
{
    public class GaDualSorter
    {
        public GaDualSorter(GenomePool<GenomeDualSorter> genomePoolDualSorter,
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

        public GenomePool<GenomeDualSorter> GenomePoolDualSorter { get; }
    }


    public static class GaDualSorterExt
    {
        public static SortingResults Eval(this GaDualSorter gaDualSorter, 
            bool saveSortResults)
        {
            var srs = gaDualSorter.SortablePool
                .AsParallel()
                .SelectMany(
                    sb => gaDualSorter.SorterPool.Select(st => st.Sort(sb)));

            return new SortingResults(
                sortResults:srs, 
                saveSortResults: saveSortResults);
        }


        public static GaDualSorter EvolveSorters(this GaDualSorter gaDualSorter,
            Dictionary<Guid, SorterResult> sorterResults,
            IRando randy, 
            int selectionFactor)
        {
            var winSortersCount = gaDualSorter.SorterPool.Count() / selectionFactor;
            var bestSorterResults = sorterResults.Values
                                       .OrderBy(r => r.AverageSortedness)
                                       .Take(winSortersCount)
                                       .ToList();
            
            var bestGenomes = bestSorterResults.GroupBy(s => s.Sorter.GenomeId)
                .Select(g => gaDualSorter.GenomePoolDualSorter.SorterGenomes[g.Key]);

            var newGenomes = bestGenomes.SelectMany(g =>
                Enumerable.Range(0, selectionFactor).Select(i => g.Mutate(randy)));

            return new GaDualSorter(
                genomePoolDualSorter: newGenomes.ToGenomePoolDualSorter(Guid.NewGuid()),
                sortablePool: gaDualSorter.SortablePool,
                randy: randy);
        }


        public static GaDualSorter EvolveSortersRecomb(this GaDualSorter gaDualSorter,
            Dictionary<Guid, SorterResult> sorterResults,
            IRando randy,
            int selectionFactor)
        {
            var winSortersCount = gaDualSorter.SorterPool.Count() / selectionFactor;
            var bestSorterResults = sorterResults.Values
                .OrderBy(r => r.AverageSortedness)
                .Take(winSortersCount)
                .ToList();

            var bestGenomes = bestSorterResults.GroupBy(s => s.Sorter.GenomeId)
                .Select(g => gaDualSorter.GenomePoolDualSorter.SorterGenomes[g.Key]);

            var bestPairs = bestGenomes.ToRandomPairs(randy);

            var recombies = bestPairs.SelectMany(p => p.Item1.Recombine(p.Item2, randy).Split());

            var newGenomes = recombies.SelectMany(g =>
                Enumerable.Range(0, selectionFactor).Select(i => g.Mutate(randy)));

            return new GaDualSorter(
                genomePoolDualSorter: newGenomes.ToGenomePoolDualSorter(Guid.NewGuid()),
                sortablePool: gaDualSorter.SortablePool,
                randy: randy);
        }


        public static GaDualSorter EvolveSortables(this GaDualSorter gaDualSorter,
            Dictionary<Guid, SortableResult> sortableResults,
            IRando randy, 
            double replacementRate)
        {
            var winSortablesCount = (int)(gaDualSorter.SortablePool.Count() * (1.0 - replacementRate));
            var looseSortablesCount = gaDualSorter.SortablePool.Count() - winSortablesCount;
            var newSortables = Enumerable.Range(0, looseSortablesCount)
                .Select(i => randy.ToPermutation(gaDualSorter.SorterPool.Order).ToSortable());

            var bestSortables = sortableResults.Values
                .OrderByDescending(r => r.AverageSortedness)
                .Take(winSortablesCount)
                .Select(sr => sr.Sortable);

            var newSortablePool = new SortablePool(Guid.NewGuid(), bestSortables.Concat(newSortables));

            return new GaDualSorter(
                genomePoolDualSorter: gaDualSorter.GenomePoolDualSorter,
                sortablePool: newSortablePool,
                randy: randy);
        }

    }


}
