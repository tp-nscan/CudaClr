using System;
using System.Collections.Generic;
using System.Linq;
using Utils.Sortable;

namespace Utils.Sorter
{
    public class SorterResult
    {
        public SorterResult(IEnumerable<SortResult> sortResults, bool saveResults)
        {
            var _sortResults = sortResults.ToList();
            Sorter = _sortResults[0].Sorter;
            double totNess = _sortResults.Sum(r => r.Sortedness);
            AverageSortedness = totNess / _sortResults.Count;
            StageUse = new int[Sorter.StageCount];
            foreach (var result in _sortResults)
            {
                for (var i = 0; i < Sorter.StageCount; i++)
                {
                    StageUse[i] += result.StageUse[i] ? 1 : 0;
                }
            }

            StagesUsed = StageUse.Sum(i => (i > 0) ? 1 : 0);

            if (saveResults)
            {
                SortResults = _sortResults.ToDictionary(r=>r.Sortable.Id);
            }
        }

        public double AverageSortedness { get; }

        public int[] StageUse { get; }

        public double StagesUsed { get; }

        public Dictionary<Guid, SortResult> SortResults { get; }

        public ISorter Sorter { get; }

    }


    public static class SorterResultExt
    {
        public static SorterResult TestSorterOn(this ISorter sorter,
            IEnumerable<ISortable> sortables, bool storeSortableResult)
        {
            return new SorterResult(
                sortResults: sortables.Select(sorter.Sort),
                saveResults: storeSortableResult);

        }
    }

}