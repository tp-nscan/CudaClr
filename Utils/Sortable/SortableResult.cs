using System;
using System.Collections.Generic;
using System.Linq;
using Utils.Sorter;

namespace Utils.Sortable
{
    public class SortableResult
    {
        public SortableResult(IEnumerable<SortResult> SortResults, bool storeSortableResult)
        {
            var _sortResults = SortResults.ToList();
            Sorter = _sortResults[0].Sorter;
            Sortable = _sortResults[0].Sortable;
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
            if (storeSortableResult)
            {
                this.SortResults = _sortResults.ToDictionary(r => r.Sorter.Id);
            }
        }

        public double AverageSortedness { get; }

        public int[] StageUse { get; }

        public Dictionary<Guid, SortResult> SortResults { get; }

        public ISortable Sortable { get; }

        public ISorter Sorter { get; }
    }
}