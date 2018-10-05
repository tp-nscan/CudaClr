using System.Collections.Generic;

namespace Utils.Ga.Parts
{
    public class SortingGaData
    {
        public SortingGaData(SorterGaResultType sorterGaResultType,
            Dictionary<string, object> data)
        {
            SorterGaResultType = sorterGaResultType;
            Data = data;
        }

        public SorterGaResultType SorterGaResultType { get; }

        public Dictionary<string, object> Data { get; }
    }

}