using System;
using System.Collections.Generic;
using System.Linq;
using Utils;
using Utils.Ga.Parts;
using Utils.Genome;
using Utils.Sortable;
using Utils.Sorter;

namespace Utils.Ga
{
    public static class GaStageDimerExt
    {
        public static GaSortingData ToRandomStageDimerGaData(
            this IRando randy, uint order,
            uint sorterCount, uint sortableCount, uint stageCount,
            double sorterWinRate, double sortableWinRate)
        {
            var randomSortablePool = randy.ToRandomSortablePool(order, sortableCount);
            var dimerGenomePool = randy.ToSorterStageDimerGenomePool(order, stageCount, sorterCount);

            var d = new Dictionary<string, object>();
            d.SetCurrentStep(0);
            d.SetSeed(randy.NextInt());
            d.SetSorterWinRate(sorterWinRate);
            d.SetSortableWinRate(sortableWinRate);
            d.SetSortablePool(randomSortablePool);
            d.SetDimerGenomePool(dimerGenomePool);

            return new GaSortingData(d);
        }

        public static GaSortingData EvolveStageDimerSorters(this GaSortingData sortingGaData,
            IRando randy)
        {
            return sortingGaData
                    .MakeSortersFromStageDimerGenomes()
                    .Eval()
                    .SelectWinningSorters()
                    .SelectSorterDimerGenomes()
                    .EvolveSorterDimerGenomes(randy);
        }


        public static GaSortingData EvolveStageDimerSortersAndSortables(this GaSortingData sortingGaData,
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


        public static GaSortingData EvolveSorterStageDimerConjRecomb_SortableConj(this GaSortingData sortingGaData,
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

        public static GaSortingData EvolveRecombineFineStageDimerSortersAndSortables(this GaSortingData sortingGaData,
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

        public static GaSortingData MakeSortersFromStageDimerGenomes(this GaSortingData sortingGaData)
        {
            var data = sortingGaData.Data.Copy();

            var dimerGenomePool = data.GetDimerGenomePool();
            var sorters = dimerGenomePool.SorterGenomes.Select(g=>g.Value.ToSorter());

            data.SetSorterPool(new SorterPool(Guid.NewGuid(), sorters));
            return new GaSortingData(data: data);
        }

        public static GaSortingData SelectSorterDimerGenomes(this GaSortingData sortingGaData)
        {
            var data = sortingGaData.Data.Copy();

            var dimerGenomePool = data.GetDimerGenomePool();
            var bestSorterPool = data.GetBestSorterPool();

            var bestDimerGenomePool = bestSorterPool
                .Select(s => dimerGenomePool.SorterGenomes[s.GenomeId])
                .ToGenomePoolStageDimer(Guid.NewGuid());

            data.SetBestDimerGenomePool(bestDimerGenomePool);
            return new GaSortingData(data: data);
        }

        public static GaSortingData RecombineSelectedSorterDimerGenomes(this GaSortingData sortingGaData, IRando randy)
        {
            var data = sortingGaData.Data.Copy();

            var bestDimerGenomePool = data.GetBestDimerGenomePool();
            data.SetBestDimerGenomePool(bestDimerGenomePool.ToRecombCoarse(randy));
            return new GaSortingData(data: data);
        }

        public static GaSortingData RecombineFineSelectedSorterDimerGenomes(this GaSortingData sortingGaData, IRando randy)
        {
            var data = sortingGaData.Data.Copy();

            var bestDimerGenomePool = data.GetBestDimerGenomePool();
            data.SetBestDimerGenomePool(bestDimerGenomePool.ToRecombFine(randy));
            return new GaSortingData(data: data);
        }

        public static GaSortingData EvolveSorterDimerGenomes(this GaSortingData sortingGaData, IRando rando)
        {
            var data = sortingGaData.Data.Copy();

            var sorterWinRate = data.GetSorterWinRate();
            var dimerGenomePool = data.GetDimerGenomePool();
            var bestDimerGenomePool = data.GetBestDimerGenomePool();
            var bestSorterPool = data.GetBestSorterPool();
            var sorterMutantCount = (int)(dimerGenomePool.SorterGenomes.Count * (1.0 - sorterWinRate));

            var newDimerGenomePool = bestDimerGenomePool
                .SorterGenomes.Values.ToRoundRobin()
                .Take(sorterMutantCount)
                .Select(g => g.Mutate(rando))
                .Concat(bestDimerGenomePool.SorterGenomes.Values)
                .ToGenomePoolStageDimer(Guid.NewGuid());

            data.SetDimerGenomePool(newDimerGenomePool);
            return new GaSortingData(data: data);
        }
    }
}
