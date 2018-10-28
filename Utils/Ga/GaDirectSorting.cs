using System;
using System.Collections.Generic;
using System.Data;
using System.Linq;
using Utils.Ga.Parts;
using Utils.Sortable;
using Utils.Sorter;

namespace Utils.Ga
{
    //public class GaDirectSortingData : GaSortingData
    //{
    //    public GaDirectSortingData(Dictionary<string, object> data) : base(data)
    //    {
    //    }
    //}

    public static class GaDirectSortingExt
    {
        public static GaData ToDirectGaSortingData(
            this IRando randy, uint order, 
            uint sorterCount, uint sortableCount, uint stageCount, 
            double sorterWinRate, double sortableWinRate, 
            StageReplacementMode stageReplacementMode)
        {
            var randomSortablePool = randy.ToRandomSortablePool(order, sortableCount);
            var randomSorterPool = randy.ToRandomSorterPool(order, stageCount, sorterCount);

            var d = new Dictionary<string, object>();
            d.SetCurrentStep(0);
            d.SetSeed(randy.NextInt());
            d.SetSorterWinRate(sorterWinRate);
            d.SetSortableWinRate(sortableWinRate);
            d.SetSortablePool(randomSortablePool);
            d.SetSorterPool(randomSorterPool);
            d.SetStageReplacementMode(stageReplacementMode);

            return new GaData(d);
        }


        public static GaData EvolveSortersConjSortablesReplace(
            this GaData sortingGaData, IRando randy)
        {
            return sortingGaData.Eval()
                                .SelectWinningSortables()
                                .SelectWinningSorters()
                                .UpdateSortablesDirect(randy)
                                .UpdateSortersDirect(randy);
        }

        public static GaData EvolveSortablesConjSortersConj(
            this GaData sortingGaData, IRando randy)
        {
            return sortingGaData.Eval()
                .SelectWinningSortables()
                .SelectWinningSorters()
                .EvolveSortablesConj(randy)
                .UpdateSortersDirect(randy);
        }

        public static GaData EvolveSortersConjRecombSortablesConj(
            this GaData sortingGaData, IRando randy)
        {
            return sortingGaData.Eval()
                .SelectWinningSortables()
                .SelectWinningSorters()
                .RecombineSelectedSorters(randy)
                .EvolveSortablesConj(randy)
                .UpdateSortersDirect(randy);
        }

        public static GaData EvolveSortablesDirect(
            this GaData sortingGaData, IRando randy)
        {
            return sortingGaData.Eval()
                                .SelectWinningSortables()
                                .UpdateSortablesDirect(randy);
        }


        public static GaData EvolveSortersDirect(
            this GaData sortingGaData, IRando randy)
        {
            return sortingGaData.Eval()
                                .SelectWinningSorters()
                                .UpdateSortersDirect(randy);
        }


        public static GaData UpdateSortersDirect(
            this GaData sortingGaData, IRando randy)
        {
            var data = sortingGaData.Data.Copy();

            var sorterWinRate = data.GetSorterWinRate();
            var wholeSorterPool = data.GetSorterPool();
            var bestSorterPool = data.GetBestSorterPool();
            var stageReplacementMode = data.GetStageReplacementMode();
            var sorterMutantCount = (int)(wholeSorterPool.Sorters.Count * (1.0 - sorterWinRate));

            var sorterNextGen = bestSorterPool.Sorters.Values.Concat(
                bestSorterPool.Sorters.Values.ToRoundRobin().Take(sorterMutantCount)
                    .Select(s => s.Mutate(randy, stageReplacementMode))
            );

            data.SetSorterPool(new SorterPool(Guid.NewGuid(), sorterNextGen));
            return new GaData(data: data);
        }

        public static GaData RecombineSelectedSorters(
            this GaData sortingGaData, IRando randy)
        {
            var data = sortingGaData.Data.Copy();

            var bestSorterPool = data.GetBestSorterPool();
            data.SetBestSorterPool(bestSorterPool.ToRecombo(randy));
            return new GaData(data: data);
        }

        public static GaData UpdateSortablesDirect(
            this GaData sortingGaData, IRando randy)
        {
            var data = sortingGaData.Data.Copy();
            var sortableWinRate = data.GetSortableWinRate();
            var bestSortablePool = data.GetBestSortablePool();
            var wholeSortablePool = data.GetSortablePool();
            var sortableMutantCount = (uint)(wholeSortablePool.Sortables.Count * (1.0 - sortableWinRate));

            var sortableNextGen = bestSortablePool.Sortables.Values.Concat(
                0u.CountUp(sortableMutantCount)
                  .Select(s => randy.ToPermutation(bestSortablePool.Order).ToSortable())
            );

            data.SetSortablePool(new SortablePool(Guid.NewGuid(), sortableNextGen));
            return new GaData(data: data);
        }

        public static GaData EvolveSortablesConj(this GaData sortingGaData, IRando randy)
        {
            var data = sortingGaData.Data.Copy();
            var sortableWinRate = data.GetSortableWinRate();
            var bestSortablePool = data.GetBestSortablePool();
            var wholeSortablePool = data.GetSortablePool();
            var sortableMutantCount = (uint)(wholeSortablePool.Sortables.Count * (1.0 - sortableWinRate));

            var sortableNextGen = bestSortablePool.Sortables.Values.Concat(
                    bestSortablePool.Sortables.Values.ToRoundRobin().Take((int)sortableMutantCount)
                        .Select(p=>p.ConjugateByRandomSingleTwoCycle(randy).ToSortable()));

            data.SetSortablePool(new SortablePool(Guid.NewGuid(), sortableNextGen));
            return new GaData(data: data);
        }


    }

}