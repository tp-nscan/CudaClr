using System;
using System.Data;
using System.Linq;
using Utils.Ga.Parts;
using Utils.Sortable;
using Utils.Sorter;

namespace Utils.Ga
{

    public static class GaSortingExt
    {
        public static GaSortingData Eval(this GaSortingData sortingGaData)
        {
            var data = sortingGaData.Data.Copy();

            var sorterPool = data.GetSorterPool();
            var sortablePool = data.GetSortablePool();

            var sr = sortablePool.AsParallel()
                                 .SelectMany(
                    sb => sorterPool.Select(st => st.Sort(sb)));

            var sortingResults = new SortingResults(sr, false);
            data.SetSortingResults(sortingResults);

            return new GaSortingData(data: data);
        }


        public static GaSortingData SelectWinningSortables(this GaSortingData sortingGaData)
        {
            var data = sortingGaData.Data.Copy();
            var step = data.GetCurrentStep();
            
            var sortablePool = data.GetSortablePool();

            var sorterWinRate = data.GetSorterWinRate();
            var sortableWinRate = data.GetSortableWinRate();
            
            var sortableWinCount = (int)(sortablePool.Sortables.Count * sortableWinRate);

            var sortingResults = data.GetSortingResults();

            var bestSortables = sortingResults.SortableResults.Values
                .OrderByDescending(r => r.AverageSortedness)
                .Take(sortableWinCount).Select(sr => sr.Sortable);
            data.SetBestSortablePool(new SortablePool(Guid.NewGuid(), bestSortables));

            return new GaSortingData(data: data);
        }


        public static GaSortingData SelectWinningSorters(this GaSortingData sortingGaData)
        {
            var data = sortingGaData.Data.Copy();

            var sorterPool = data.GetSorterPool();
            var sorterWinRate = data.GetSorterWinRate();
            var sorterWinCount = (int)(sorterPool.Sorters.Count * sorterWinRate);
            var sortingResults = data.GetSortingResults();
            var bestSorters = sortingResults.SorterResults.Values
                .OrderBy(r => r.AverageSortedness)
                .Take(sorterWinCount).Select(sr => sr.Sorter);

            data.SetBestSorterPool(new SorterPool(Guid.NewGuid(), bestSorters));

            return new GaSortingData(data: data);
        }


        public static string Report(this GaSortingData sortingGaData)
        {
            var data = sortingGaData.Data;
            var bestSorterPool = data.GetBestSorterPool();
            var sortingResults = data.GetSortingResults();

            var avgAllSorters = sortingResults.SorterResults.Average(sr => sr.Value.AverageSortedness);

            return $"{avgAllSorters}";
        }


        public static string ReportMore(this GaSortingData sortingGaData)
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


        public static string CompareReport(this GaSortingData sgdNew, GaSortingData sgdOld)
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


        public static string CompareReportLarge(this GaSortingData sgdNew, GaSortingData sgdOld)
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