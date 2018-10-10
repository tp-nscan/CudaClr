using System;
using System.Collections.Generic;

namespace Utils
{
    public class ProcResult
    {
        public ProcResult(Dictionary<string, object> data, string err = "", 
            int stepsCompleted = 0, double time = 0.0)
        {
            Data = data;
            ErrorMsg = err;
            StepsCompleted = stepsCompleted;
            TimeInMs = time;
        }

        public Dictionary<string, object> Data { get; private set; }
        public string ErrorMsg { get; private set; }
        public int StepsCompleted { get; private set; }
        public double TimeInMs { get; private set; }

    }

}
