namespace Utils
{
    public interface ISorter
    {
        ISorterStage this[int index] { get; }
        int SorterStageCount { get; }
    }

    public class SorterResult
    {
        public SorterResult(bool sucessful, int[] stageUse, IPermutation input, IPermutation output)
        {
            Sucessful = sucessful;
            StageUse = stageUse;
            Input = input;
            Output = output;
        }

        public bool Sucessful { get; }

        public int[] StageUse { get; }

        public IPermutation Input { get; }

        public IPermutation Output { get; }

    }

    public static class SorterEx
    {
        public static SorterResult Sort(this ISorter sorter, IPermutation perm)
        {

            return new SorterResult(
                sucessful:true, stageUse: new int[] { }, input:null, output:null);
        }

    }
}