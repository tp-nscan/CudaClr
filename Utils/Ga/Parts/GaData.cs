using System.Collections.Generic;

namespace Utils.Ga.Parts
{
    public class GaData
    {
        public GaData(Dictionary<string, object> data)
        {
            Data = data;
        }
        public Dictionary<string, object> Data { get; private set; }
    }

}