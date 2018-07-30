using System.Collections.Generic;

namespace Sponge.Model
{
    public class ProcResult
    {
        public ProcResult(Dictionary<string, object> data, string err, 
            int steps, double time)
        {
            Data = data;
            ErrorMsg = err;
            StepsCompleted = steps;
            TimeInMs = time;
        }

        public Dictionary<string, object> Data { get; private set; }
        public string ErrorMsg { get; private set; }
        public int StepsCompleted { get; private set; }
        public double TimeInMs { get; private set; }

    }

}
