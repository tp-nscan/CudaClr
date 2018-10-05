using System;
using System.Collections.Generic;
using System.Data;
using System.Linq;
using Utils.Ga.Parts;
using Utils.Sortable;
using Utils.Sorter;

namespace Utils.Ga
{
    public static class GaDirectSortingExt
    {
        public static SortingGaData ToRandomDirectSortingGaData(
            this IRando randy, uint order, 
            uint sorterCount, uint sortableCount, uint stageCount, 
            double sorterWinRate, double sortableWinRate, 
            StageReplacementMode stageReplacementMode)
        {
            var randomSortablePool = randy.ToRandomSortablePool(order, sortableCount);
            var randomSorterPool = randy.ToRandomSorterPool(order, stageCount, sorterCount);

            var d = new Dictionary<string, object>();
            d.SetCurrentStep(1);
            d.SetSorterWinRate(sorterWinRate);
            d.SetSortableWinRate(sortableWinRate);
            d.SetSortablePool(randomSortablePool);
            d.SetSorterPool(randomSorterPool);
            d.SetStageReplacementMode(stageReplacementMode);

            return new SortingGaData(SorterGaResultType.Normal, d);
        }


        public static SortingGaData EvolveBothDirect(this SortingGaData sortingGaData,
            IRando randy)
        {
            return sortingGaData.Eval()
                                .SelectWinningSortables()
                                .SelectWinningSorters()
                                .UpdateSortablesDirect(randy)
                                .UpdateSortersDirect(randy);
        }


        public static SortingGaData EvolveSortablesDirect(this SortingGaData sortingGaData,
            IRando randy)
        {
            return sortingGaData.Eval()
                                .SelectWinningSortables()
                                .UpdateSortablesDirect(randy);
        }


        public static SortingGaData EvolveSortersDirect(this SortingGaData sortingGaData,
            IRando randy)
        {
            return sortingGaData.Eval()
                                .SelectWinningSorters()
                                .UpdateSortersDirect(randy);
        }


        public static SortingGaData UpdateSortersDirect(this SortingGaData sortingGaData, IRando randy)
        {
            var data = sortingGaData.Data.Copy();
            data.SetCurrentStep(data.GetCurrentStep() + 1);

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
            return new SortingGaData(
                sorterGaResultType: SorterGaResultType.Normal,
                data: data
            );
        }


        public static SortingGaData UpdateSortablesDirect(this SortingGaData sortingGaData, IRando randy)
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
            return new SortingGaData(
                sorterGaResultType: SorterGaResultType.Normal,
                data: data
            );
        }

    }

}