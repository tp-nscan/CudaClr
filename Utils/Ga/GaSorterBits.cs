﻿using System;
using System.Collections.Generic;
using System.Linq;
using Utils.Genome;
using Utils.Sortable;
using Utils.Sorter;

namespace Utils.Ga
{
    public class GaSorterBits
    {
        public GaSorterBits(
                            GenomePoolSorterBits genomePoolSorterBits,
                            SortablePool sortablePool,
                            IRando randy)
        {
            SortablePool = sortablePool;
            GenomePoolSorterBits = genomePoolSorterBits;

            SorterPool = new SorterPool(
                id: Guid.NewGuid(),
                sorters: GenomePoolSorterBits.Select(
                         gds => gds.MakePhenotype(randy)));
        }

        public SorterPool SorterPool { get; }

        public SortablePool SortablePool { get; }

        public GenomePoolSorterBits GenomePoolSorterBits { get; }
    }


    public class GaSorterBitsResult
    {
        public GaSorterBitsResult(GaSorterBits gaSorterBits,
            IEnumerable<SortResult> sortResults, bool saveSortResults)
        {
            GaSorterBits = gaSorterBits;

            var srl = sortResults.ToList();

            var gbSorter = srl.GroupBy(sr => sr.Sorter).ToList();
            SorterResults = gbSorter.Select(
                g => new SorterResult(g, saveSortResults)).ToDictionary(g => g.Sorter.Id);

            var gbSortable = srl.GroupBy(sr => sr.Input).ToList();
            SortableResults = gbSortable.Select(
                g => new SortableResult(g, saveSortResults)).ToDictionary(g => g.Sortable.Id);

        }

        public GaSorterBits GaSorterBits { get; }

        public Dictionary<Guid, SorterResult> SorterResults { get; }

        public Dictionary<Guid, SortableResult> SortableResults { get; }
    }


    public static class GaSorterBitsExt
    {
        public static GaSorterBitsResult Eval(this GaSorterBits gaSorterBits,
            bool saveSortResults)
        {
            var srs = gaSorterBits.SortablePool
                .AsParallel()
                .SelectMany(
                    sb => gaSorterBits.SorterPool.Select(st => st.Sort(sb)));

            return new GaSorterBitsResult(
                gaSorterBits: gaSorterBits,
                sortResults: srs,
                saveSortResults: saveSortResults);
        }

        public static GaSorterBits EvolveSorters(this GaSorterBitsResult res, IRando randy, 
            int selectionFactor, double mutationRate)
        {
            var winSortersCount = res.GaSorterBits.SorterPool.Count() / selectionFactor;
            var bestSorterResults = res.SorterResults.Values
                                       .OrderBy(r => r.AverageSortedness)
                                       .Take(winSortersCount)
                                       .ToList();

            var bestGenomes = bestSorterResults.GroupBy(s => s.Sorter.GenomeId)
                .Select(g => res.GaSorterBits.GenomePoolSorterBits.GenomeSorterBitses[g.Key]);

            var newGenomes = bestGenomes.Concat(bestGenomes.SelectMany(g =>
                Enumerable.Range(0, selectionFactor - 1).Select(i => g.Mutate(id:Guid.NewGuid(), 
                    randy: randy, mutationRate: mutationRate))));

            return new GaSorterBits(
                genomePoolSorterBits: new GenomePoolSorterBits(Guid.NewGuid(), newGenomes),
                sortablePool: res.GaSorterBits.SortablePool,
                randy: randy);
        }

        //public static GaDualSorter EvolveSortersRecomb(this GaDualSorterResult res, IRando randy, int selectionFactor)
        //{
        //    var winSortersCount = res.GaDualSorter.SorterPool.Count() / selectionFactor;
        //    var bestSorterResults = res.SorterResults.Values
        //        .OrderBy(r => r.AverageSortedness)
        //        .Take(winSortersCount)
        //        .ToList();

        //    var bestGenomes = bestSorterResults.GroupBy(s => s.Sorter.GenomeId)
        //        .Select(g => res.GaDualSorter.GenomePoolDualSorter.GenomeDualSorters[g.Key]);

        //    var bestPairs = bestGenomes.ToRandomPairs(randy);

        //    var recombies = bestPairs.SelectMany(p => p.Item1.Recombine(p.Item2, randy).Split());

        //    var newGenomes = recombies.SelectMany(g =>
        //        Enumerable.Range(0, selectionFactor).Select(i => g.Mutate(randy)));

        //    return new GaDualSorter(
        //        genomePoolDualSorter: new GenomePoolDualSorter(Guid.NewGuid(), newGenomes),
        //        sortablePool: res.GaDualSorter.SortablePool,
        //        randy: randy);
        //}


        public static GaSorterBits EvolveSortables(this GaSorterBitsResult res, IRando randy, double replacementRate)
        {
            var winSortablesCount = (int)(res.GaSorterBits.SortablePool.Count() * (1.0 - replacementRate));
            var looseSortablesCount = res.GaSorterBits.SortablePool.Count() - winSortablesCount;
            var newSortables = Enumerable.Range(0, looseSortablesCount)
                .Select(i => randy.ToPermutation(res.GaSorterBits.SorterPool.Order).ToSortable());

            var bestSortables = res.SortableResults.Values
                .OrderByDescending(r => r.AverageSortedness)
                .Take(winSortablesCount)
                .Select(sr => sr.Sortable);

            var newSortablePool = new SortablePool(Guid.NewGuid(), bestSortables.Concat(newSortables));

            return new GaSorterBits(
                genomePoolSorterBits: res.GaSorterBits.GenomePoolSorterBits,
                sortablePool: newSortablePool,
                randy: randy);
        }

    }


}