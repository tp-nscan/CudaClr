using Microsoft.VisualStudio.TestTools.UnitTesting;
using Utils;
using Utils.Sorter;

namespace SpongeTester
{
    [TestClass]
    public class SorterPoolFixture
    {

        [TestMethod]
        public void TestToSorterDistr()
        {
            var seed = 1223;
            const int order = 5;
            const int stageCount = 2;
            const int sorterCount = 1000;

            var randy = Rando.Standard(seed);
            var p1 = randy.ToRandomSorterPool(order, stageCount, sorterCount);

            var distr = p1.ToSorterDistr();
        }

        [TestMethod]
        public void TestToRecombo()
        {
            var seed = 1223;
            const int order = 5;
            const int stageCount = 2;
            const int sorterCount = 1000;

            var randy = Rando.Standard(seed);
            var p1 = randy.ToRandomSorterPool(order, stageCount, sorterCount);

            var pp = p1.ToRecombo(randy);
        }



    }
}
