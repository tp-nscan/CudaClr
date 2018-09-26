using Utils.Sortable;

namespace Utils.Sorter
{
    public class SortResult
    {
        public SortResult(ISorter sorter, uint sortedness, bool[] stageUse,
            ISortable sortable, IPermutation output)
        {
            Sortedness = sortedness;
            StageUse = stageUse;
            Sortable = sortable;
            Output = output;
            Sorter = sorter;
        }

        public uint Sortedness { get; }

        public bool[] StageUse { get; }

        public ISortable Sortable { get; }

        public ISorter Sorter { get; }

        public IPermutation Output { get; }

    }
}