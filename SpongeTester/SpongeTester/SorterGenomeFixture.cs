using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Utils;

namespace SpongeTester
{
    [TestClass]
    public class SorterGenomeFixture
    {
        [TestMethod]
        public void MakeDualSorterGenome()
        {
            const int seed = 5234;
            const int order = 7;
            const int stageCount = 2;

            var randy = Rando.Standard(seed);
            var dsg = randy.ToDualSorterGenome(order, stageCount);

            var mutant = dsg.Mutate(randy);
        }

        [TestMethod]
        public void MakeDualSorterGenomePool()
        {
            const int seed = 5234;
            const int order = 7;
            const int stageCount = 2;
            const int poolCount = 2;

            var randy = Rando.Standard(seed);
            var dsg = randy.ToGenomePoolDualSorter(order, stageCount, poolCount);
            
        }




    }
}
