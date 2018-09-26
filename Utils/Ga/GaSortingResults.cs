using System;
using System.Collections.Generic;
using System.Linq;
using Utils.Sortable;
using Utils.Sorter;

namespace Utils.Ga
{
    public class GaSortingResults
    {
        public GaSortingResults(
            IEnumerable<SortResult> sortResults, bool saveSortResults)
        {
            var srl = sortResults.ToList();

            var gbSorter = srl.GroupBy(sr => sr.Sorter).ToList();
            SorterResults = gbSorter.Select(
                g => new SorterResult(g, saveSortResults)).ToDictionary(g => g.Sorter.Id);

            var gbSortable = srl.GroupBy(sr => sr.Sortable).ToList();
            SortableResults = gbSortable.Select(
                g => new SortableResult(g, saveSortResults)).ToDictionary(g => g.Sortable.Id);

        }
        
        public Dictionary<Guid, SorterResult> SorterResults { get; }

        public Dictionary<Guid, SortableResult> SortableResults { get; }
    }


}
