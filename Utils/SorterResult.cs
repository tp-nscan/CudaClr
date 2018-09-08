using System;
using System.Collections.Generic;
using System.Linq;

namespace Utils
{
    public class SorterResult
    {
        public SorterResult(IEnumerable<SortResult> SortResults, bool saveResults)
        {
            var _sortResults = SortResults.ToList();
            Sorter = _sortResults[0].Sorter;
            float totNess = _sortResults.Sum(r => r.Sortedness);
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
                this.SortResults = _sortResults.ToDictionary(r=>r.Input.Id);
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
                SortResults: sortables.Select(sorter.Sort),
                saveResults: storeSortableResult);

        }
    }

}