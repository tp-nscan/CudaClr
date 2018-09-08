namespace Utils
{
    public class SortResult
    {
        public SortResult(ISorter sorter, int sortedness, bool[] stageUse,
            ISortable input, IPermutation output)
        {
            Sortedness = sortedness;
            StageUse = stageUse;
            Input = input;
            Output = output;
            Sorter = sorter;
        }

        public int Sortedness { get; }

        public bool[] StageUse { get; }

        public ISortable Input { get; }

        public ISorter Sorter { get; }

        public IPermutation Output { get; }

    }
}