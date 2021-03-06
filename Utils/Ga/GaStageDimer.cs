﻿using System;
using System.Collections.Generic;
using System.Linq;
using Utils;
using Utils.Ga.Parts;
using Utils.Genome;
using Utils.Genome.Sorter;
using Utils.Sortable;
using Utils.Sorter;

namespace Utils.Ga
{
    public static class GaStageDimerExt
    {

        public static GaData ToStageDimerGaData(
            this IRando randy, uint order,
            uint sorterCount, uint sortableCount, uint stageCount,
            double sorterWinRate, double sortableWinRate)
        {
            var randomSortablePool = randy.ToRandomSortablePool(order, sortableCount);
            var dimerGenomePool = randy.ToGenomePoolStageDimer(order, stageCount, sorterCount);

            var d = new Dictionary<string, object>();
            d.SetCurrentStep(0);
            d.SetSeed(randy.NextInt());
            d.SetSorterWinRate(sorterWinRate);
            d.SetSortableWinRate(sortableWinRate);
            d.SetSortablePool(randomSortablePool);
            d.SetDimerGenomePool(dimerGenomePool);

            return new GaData(d);
        }

        public static GaData EvolveStageDimerSorters(this GaData sortingGaData,
            IRando randy)
        {
            return sortingGaData
                    .MakeSortersFromStageDimerGenomes()
                    .Eval()
                    .SelectWinningSorters()
                    .SelectSorterDimerGenomes()
                    .EvolveSorterDimerGenomes(randy);
        }


        public static GaData EvolveStageDimerSortersAndSortables(this GaData sortingGaData,
            IRando randy)
        {
            return sortingGaData
                    .MakeSortersFromStageDimerGenomes()
                    .Eval()
                    .SelectWinningSorters()
                    .SelectWinningSortables()
                    .SelectSorterDimerGenomes()
                    .SelectWinningSortables()
                    .EvolveSortablesConj(randy)
                    .EvolveSorterDimerGenomes(randy);
        }


        public static GaData EvolveSorterStageDimerConjRecomb_SortableConj(this GaData sortingGaData,
            IRando randy)
        {
            return sortingGaData
                    .MakeSortersFromStageDimerGenomes()
                    .Eval()
                    .SelectWinningSorters()
                    .SelectWinningSortables()
                    .SelectSorterDimerGenomes()
                    .RecombineSelectedSorterDimerGenomes(randy)
                    .SelectWinningSortables()
                    .EvolveSortablesConj(randy)
                    .EvolveSorterDimerGenomes(randy);
        }

        public static GaData EvolveRecombineFineStageDimerSortersAndSortables(this GaData sortingGaData,
            IRando randy)
        {
            return sortingGaData
                .MakeSortersFromStageDimerGenomes()
                .Eval()
                .SelectWinningSorters()
                .SelectWinningSortables()
                .SelectSorterDimerGenomes()
                .RecombineFineSelectedSorterDimerGenomes(randy)
                .SelectWinningSortables()
                .EvolveSortablesConj(randy)
                .EvolveSorterDimerGenomes(randy);
        }

        public static GaData MakeSortersFromStageDimerGenomes(
            this GaData sortingGaData)
        {
            var data = sortingGaData.Data.Copy();

            var dimerGenomePool = data.GetDimerGenomePool();
            var sorters = dimerGenomePool.SorterGenomes.Select(g=>g.Value.ToSorter());

            data.SetSorterPool(new SorterPool(Guid.NewGuid(), sorters));
            return new GaData(data: data);
        }

        public static GaData SelectSorterDimerGenomes(
            this GaData sortingGaData)
        {
            var data = sortingGaData.Data.Copy();

            var dimerGenomePool = data.GetDimerGenomePool();
            var bestSorterPool = data.GetBestSorterPool();

            var bestDimerGenomePool = bestSorterPool
                .Select(s => dimerGenomePool.SorterGenomes[s.GenomeId])
                .ToGenomePoolStageDimer(Guid.NewGuid());

            data.SetBestDimerGenomePool(bestDimerGenomePool);
            return new GaData(data: data);
        }

        public static GaData RecombineSelectedSorterDimerGenomes(
            this GaData sortingGaData, IRando randy)
        {
            var data = sortingGaData.Data.Copy();

            var bestDimerGenomePool = data.GetBestDimerGenomePool();
            data.SetBestDimerGenomePool(bestDimerGenomePool.ToRecombCoarse(randy));
            return new GaData(data: data);
        }

        public static GaData RecombineFineSelectedSorterDimerGenomes(
            this GaData sortingGaData, IRando randy)
        {
            var data = sortingGaData.Data.Copy();

            var bestDimerGenomePool = data.GetBestDimerGenomePool();
            data.SetBestDimerGenomePool(bestDimerGenomePool.ToRecombFine(randy));
            return new GaData(data: data);
        }

        public static GaData EvolveSorterDimerGenomes(this GaData sortingGaData, IRando rando)
        {
            var data = sortingGaData.Data.Copy();

            var sorterWinRate = data.GetSorterWinRate();
            var dimerGenomePool = data.GetDimerGenomePool();
            var bestDimerGenomePool = data.GetBestDimerGenomePool();
            var sorterMutantCount = (int)(dimerGenomePool.SorterGenomes.Count * (1.0 - sorterWinRate));

            var newDimerGenomePool = bestDimerGenomePool
                .SorterGenomes.Values.ToRoundRobin()
                .Take(sorterMutantCount)
                .Select(g => g.Mutate(rando))
                .Concat(bestDimerGenomePool.SorterGenomes.Values)
                .ToGenomePoolStageDimer(Guid.NewGuid());

            data.SetDimerGenomePool(newDimerGenomePool);
            return new GaData(data: data);
        }
    }
}
