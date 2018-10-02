using System;
using System.Collections.Generic;
using System.Data;
using System.Linq;
using Utils.Sortable;
using Utils.Sorter;

namespace Utils.Ga
{
    public interface ISortingGaRunner
    {
        uint StepNumber { get; }
        SortingGaData SortingGaData { get; }
        ISortingGaRunner NextStep(ISortingGaRunner sortingGaRunner);
    }


    public static class DirectSortingGaExt
    {
        public static SortingGaData ToDirectRandomSortingGaData(this IRando randy, 
            uint order, uint sorterCount, uint sortableCount, uint stageCount, 
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


        public static SortingGaData EvolveBoth(this SortingGaData sortingGaData,
            IRando randy)
        {
            return sortingGaData.Eval().SelectWinners().UpdateSortables(randy)
                                .UpdateSorters(randy);
        }

        public static SortingGaData EvolveJustSortables(this SortingGaData sortingGaData,
            IRando randy)
        {
            return sortingGaData.Eval().SelectWinners().UpdateSortables(randy);
        }

        public static SortingGaData EvolveJustSorters(this SortingGaData sortingGaData,
            IRando randy)
        {
            return sortingGaData.Eval().SelectWinners().UpdateSorters(randy);
        }

        public static SortingGaData Eval(this SortingGaData sortingGaData)
        {
            var data = sortingGaData.Data.Copy();

            var sorterPool = data.GetSorterPool();
            var sortablePool = data.GetSortablePool();

            var sr = sortablePool.AsParallel()
                                 .SelectMany(
                    sb => sorterPool.Select(st => st.Sort(sb)));

            var sortingResults = new SortingResults(sr, false);
            data.SetSortingResults(sortingResults);

            return new SortingGaData(
                sorterGaResultType: SorterGaResultType.Normal,
                data: data
            );
        }

        public static SortingGaData SelectWinners(this SortingGaData sortingGaData)
        {
            var data = sortingGaData.Data.Copy();
            var step = data.GetCurrentStep();
            //var mode = data.GetDirectSortingGaRunner_Mode();
            data.SetCurrentStep(1);

            var sorterPool = data.GetSorterPool();
            var sortablePool = data.GetSortablePool();

            var sorterWinRate = data.GetSorterWinRate();
            var sortableWinRate = data.GetSortableWinRate();

            var sorterWinCount = (int)(sorterPool.Sorters.Count * sorterWinRate);
            var sortableWinCount = (int)(sortablePool.Sortables.Count * sortableWinRate);

            var sortingResults = data.GetSortingResults();

            var bestSorters = sortingResults.SorterResults.Values
                .OrderBy(r => r.AverageSortedness)
                .Take(sorterWinCount).Select(sr=>sr.Sorter);

            data.SetBestSorterPool(new SorterPool(Guid.NewGuid(), bestSorters));

            var bestSortables = sortingResults.SortableResults.Values
                .OrderByDescending(r => r.AverageSortedness)
                .Take(sortableWinCount).Select(sr => sr.Sortable);
            data.SetBestSortablePool(new SortablePool(Guid.NewGuid(), bestSortables));

            return new SortingGaData(
                    sorterGaResultType:SorterGaResultType.Normal,
                    data:data
                );
        }

        public static SortingGaData UpdateSorters(this SortingGaData sortingGaData, IRando randy)
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

        public static SortingGaData UpdateSortables(this SortingGaData sortingGaData, IRando randy)
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

        public static string Report(this SortingGaData sortingGaData)
        {
            var data = sortingGaData.Data;
            var bestSorterPool = data.GetBestSorterPool();
            var sortingResults = data.GetSortingResults();

            var avgAllSorters = sortingResults.SorterResults.Average(sr => sr.Value.AverageSortedness);

            return $"{avgAllSorters}";
        }


        public static string ReportMore(this SortingGaData sortingGaData)
        {
            var data = sortingGaData.Data;
            var bestSorterPool = data.GetBestSorterPool();
            var sortingResults = data.GetSortingResults();

            var avgAllSorters = sortingResults.SorterResults.Average(sr => sr.Value.AverageSortedness);
            var avgBestSorters = bestSorterPool.Sorters.Values
                .Select(s => sortingResults.SorterResults[s.Id])
                .Average(sr => sr.AverageSortedness);

            var sorterDiv = data.GetBestSorterPool().Sorters.Values
                .Select(s => sortingResults.SorterResults[s.Id])
                .GroupBy(sr => sr.AverageSortedness).Count();

            return $"{avgAllSorters} {avgBestSorters} {sorterDiv}";
        }


        public static string CompareReport(this SortingGaData sgdNew, SortingGaData sgdOld)
        {
            var newData = sgdNew.Data;
            var oldData = sgdOld.Data;

            var sorterPool = newData.GetSorterPool();
            var sortablePool = oldData.GetSortablePool();

            var srtt = sortablePool.AsParallel()
                .SelectMany(
                    sb => sorterPool.Select(st => st.Sort(sb)));

            var sortingResults = new SortingResults(srtt, false);

            var avgAllSorters = sortingResults.SorterResults.Average(sr => sr.Value.AverageSortedness);
            var avgBestSorters = newData.GetBestSorterPool().Sorters.Values
                .Select(s => sortingResults.SorterResults[s.Id])
                .Average(sr => sr.AverageSortedness);

            return $"{avgAllSorters} ";
        }

        public static string CompareReportLarge(this SortingGaData sgdNew, SortingGaData sgdOld)
        {
            var newData = sgdNew.Data;
            var oldData = sgdOld.Data;

            var sorterPool = newData.GetSorterPool();
            var sortablePool = oldData.GetSortablePool();

            var srtt = sortablePool.AsParallel()
                .SelectMany(
                    sb => sorterPool.Select(st => st.Sort(sb)));

            var sortingResults = new SortingResults(srtt, false);
            
            var avgAllSorters = sortingResults.SorterResults.Average(sr => sr.Value.AverageSortedness);
            var avgBestSorters = newData.GetBestSorterPool().Sorters.Values
                .Select(s => sortingResults.SorterResults[s.Id])
                .Average(sr => sr.AverageSortedness);

            return $"{avgAllSorters} {avgBestSorters}";
        }

    }

}