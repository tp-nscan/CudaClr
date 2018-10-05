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
        public static SortingGaData ToRandomStageDimerGaData(
            this IRando randy, uint order,
            uint sorterCount, uint sortableCount, uint stageCount,
            double sorterWinRate, double sortableWinRate)
        {
            var randomSortablePool = randy.ToRandomSortablePool(order, sortableCount);
            var dimerGenomePool = randy.ToSorterStageDimerGenomePool(order, stageCount, sorterCount);

            var d = new Dictionary<string, object>();
            d.SetCurrentStep(1);
            d.SetSorterWinRate(sorterWinRate);
            d.SetSortableWinRate(sortableWinRate);
            d.SetSortablePool(randomSortablePool);
            d.SetDimerGenomePool(dimerGenomePool);

            return new SortingGaData(SorterGaResultType.Normal, d);
        }

        public static SortingGaData EvolveStageDimerSorters(this SortingGaData sortingGaData,
            IRando randy)
        {
            return sortingGaData.MakeSortersFromDimerGenomes()
                .Eval()
                .SelectWinningSorters()
                .SelectSorterDimerGenomes()
                .EvolveSorterDimerGenomes(randy);
        }

        public static SortingGaData MakeSortersFromDimerGenomes(this SortingGaData sortingGaData)
        {
            var data = sortingGaData.Data.Copy();

            var dimerGenomePool = data.GetDimerGenomePool();
            var sorters = dimerGenomePool.SorterGenomes.Select(g=>g.Value.ToSorter());

            data.SetSorterPool(new SorterPool(Guid.NewGuid(), sorters));
            return new SortingGaData(
                sorterGaResultType: SorterGaResultType.Normal,
                data: data
            );
        }

        public static SortingGaData SelectSorterDimerGenomes(this SortingGaData sortingGaData)
        {
            var data = sortingGaData.Data.Copy();

            var dimerGenomePool = data.GetDimerGenomePool();
            var bestSorterPool = data.GetBestSorterPool();

            var bestDimerGenomePool = bestSorterPool
                .Select(s => dimerGenomePool.SorterGenomes[s.GenomeId])
                .ToGenomePoolStageDimer(Guid.NewGuid());

            data.SetBestDimerGenomePool(bestDimerGenomePool);
            return new SortingGaData(
                sorterGaResultType: SorterGaResultType.Normal,
                data: data
            );
        }

        public static SortingGaData EvolveSorterDimerGenomes(this SortingGaData sortingGaData, IRando rando)
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
            return new SortingGaData(
                sorterGaResultType: SorterGaResultType.Normal,
                data: data
            );
        }
    }
}
