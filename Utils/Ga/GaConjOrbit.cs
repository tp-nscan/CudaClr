using System;
using System.Collections.Generic;
using System.Linq;
using Utils.Genome;
using Utils.Ga.Parts;
using Utils.Genome.Sorter;
using Utils.Sortable;
using Utils.Sorter;

namespace Utils.Ga
{
    public class GaConjOrbit : GaSortingData
    {
        public GaConjOrbit(Dictionary<string, object> data) : base(data)
        {
        }
    }

    public static class GaConjOrbitExt
    {

        public static GaSortingData ToGaConjOrbitData(
            this IRando randy, uint order,
            uint sorterCount, uint sortableCount, uint stageCount,
            double sorterWinRate, double sortableWinRate)
        {
            var randomSortablePool = randy.ToRandomSortablePool(order, sortableCount);
            var conjOrbitGenomePool = randy.ToGenomePoolConjOrbits(order, stageCount, sorterCount);

            var d = new Dictionary<string, object>();
            d.SetCurrentStep(0);
            d.SetSeed(randy.NextInt());
            d.SetSorterWinRate(sorterWinRate);
            d.SetSortablePool(randomSortablePool);
            d.SetSortableWinRate(sortableWinRate);
            d.SetConjOrbitGenomePool(conjOrbitGenomePool);

            return new GaSortingData(d);
        }


        public static GaSortingData EvolveConjOrbitSortersAndSortables(this GaSortingData sortingGaData,
            IRando randy)
        {
            //return sortingGaData
            //    .MakeSortersFromConjOrbitGenomes()
            //    .Eval()
            //    .SelectWinningSorters()
            //    .SelectWinningSortables()
            //    .SelectConjOrbitGenomes()
            //    .SelectWinningSortables()
            //    .EvolveSortablesConj(randy)
            //    .EvolveConjOrbitGenomes(randy);

            return sortingGaData
                .MakeSortersFromConjOrbitGenomes()
                .Eval()
                .SelectWinningSorters()
                .SelectConjOrbitGenomes()
                .SelectWinningSortables()
                .EvolveConjOrbitGenomes(randy);
        }

        public static GaSortingData MakeSortersFromConjOrbitGenomes(
            this GaSortingData sortingGaData)
        {
            var data = sortingGaData.Data.Copy();

            var conjOrbitGenomePool = data.GetConjOrbitGenomePool();
            var sorters = conjOrbitGenomePool
                .SorterGenomes
                .Select(g => g.Value.ToSorter(g.Value.Stagecount));

            data.SetSorterPool(new SorterPool(Guid.NewGuid(), sorters));
            return new GaSortingData(data: data);
        }


        public static GaSortingData SelectConjOrbitGenomes(
            this GaSortingData sortingGaData)
        {
            var data = sortingGaData.Data.Copy();

            var conjOrbitGenomePool = data.GetConjOrbitGenomePool();
            var bestSorterPool = data.GetBestSorterPool();

            var bestConjOrbitGenomePool = bestSorterPool
                .Select(s => conjOrbitGenomePool.SorterGenomes[s.GenomeId])
                .ToGenomePoolConjOrbits(Guid.NewGuid());

            data.SetBestConjOrbitGenomePool(bestConjOrbitGenomePool);
            return new GaSortingData(data: data);
        }

        public static GaSortingData EvolveConjOrbitGenomes(this GaSortingData sortingGaData, IRando rando)
        {
            var data = sortingGaData.Data.Copy();

            var sorterWinRate = data.GetSorterWinRate();
            var conjOrbitGenomePool = data.GetConjOrbitGenomePool();
            var bestConjOrbitGenomePool = data.GetBestConjOrbitGenomePool();
            var bestSorterPool = data.GetBestSorterPool();
            var sorterMutantCount = (int)(conjOrbitGenomePool.SorterGenomes.Count * (1.0 - sorterWinRate));

            var newDimerGenomePool = bestConjOrbitGenomePool
                .SorterGenomes.Values.ToRoundRobin()
                .Take(sorterMutantCount)
                .Select(g => g.Mutate(rando))
                .Concat(bestConjOrbitGenomePool.SorterGenomes.Values)
                .ToGenomePoolConjOrbits(Guid.NewGuid());

            data.SetConjOrbitGenomePool(newDimerGenomePool);
            return new GaSortingData(data: data);
        }

    }
}
