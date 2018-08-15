using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Utils;

namespace Test
{
    class Program
    {
        static void Main(string[] args)
        {
            TestEnergy();
        }

        public static void TestEnergy()
        {
            var grid = new int[] { -1, -1, -1, -1,  1, -1, -1, -1, -1 };

            var res = GridFuncs.k_Energy4(grid, 3);

        }
    }
}
