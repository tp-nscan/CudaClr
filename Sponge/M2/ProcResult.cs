using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Sponge.M2
{
    public class ProcResult<T>
    {
        public ProcResult(T data, string err, int steps, double time)
        {
            Data = data;
            ErrorMsg = err;
            StepsCompleted = steps;
            TimeInMs = time;
        }

        public T Data { get; private set; }
        public string ErrorMsg { get; private set; }
        public int StepsCompleted { get; private set; }
        public double TimeInMs { get; private set; }

    }
}
