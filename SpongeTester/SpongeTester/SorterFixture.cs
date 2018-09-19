using System;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Utils;

namespace SpongeTester
{
    [TestClass]
    public class SorterFixture
    {

        [TestMethod]
        public void TestSortliness()
        {
            const int order = 24;
            const int trials = 5000;
            var randy = Rando.Standard(1444);

            for (var i = 0; i < trials; i++)
            {
                var stagey = randy.RandomFullSorterStage(order, 0);
                var stagey2 = randy.RandomFullSorterStage(order, 0);
                var stagey3 = randy.RandomFullSorterStage(order, 0);
                var stagey4 = randy.RandomFullSorterStage(order, 0);

                var permy = randy.RandomPermutation(order);

                var strey0 = stagey.Sort(permy);
                var strey2 = stagey2.Sort(strey0.Item2);
                var strey3 = stagey3.Sort(strey2.Item2);
                var strey4 = stagey4.Sort(strey3.Item2);
                var strey5 = stagey.Sort(strey4.Item2);
                var strey6 = stagey2.Sort(strey5.Item2);
                var strey7 = stagey3.Sort(strey6.Item2);
                var strey8 = stagey4.Sort(strey7.Item2);

                var pr = permy.Sortedness();
                var psr0 = strey0.Item2.Sortedness();
                var psr2 = strey8.Item2.Sortedness();

                Console.WriteLine($"{pr} {psr0} {psr2}");
            }
        }


        [TestMethod]
        public void TestSorterSort()
        {
            const uint order = 24;
            var permCount = 2000u;
            var sorterCount = 200u;
            var randy = Rando.Standard(1444);

            var permies = 0u.CountUp(permCount)
                .Select(i => randy.RandomPermutation(order).ToSortable())
                .ToList();

            var sorters = 0u.CountUp(sorterCount)
                .Select(i => randy.ToSorter(order, i))
                .ToList();

            for (var i = 0; i < sorterCount; i++)
            {
                var res = permies.Select(p => sorters[i].Sort(p)).ToList();
                for (var j = 0; j < res.Count(); j++)
                {
                    Console.WriteLine($"{j} {i} {res[j].Sortedness}");
                }
            }
        }


        [TestMethod]
        public void TestReplaceStage()
        {
            const uint order = 24;
            const uint stageCount = 5;
            const uint beforeIndex = 4;
            var randy = Rando.Standard(1444);
            var oldSorter = randy.ToSorter(order, stageCount);
            var newSorter = oldSorter.Mutate(randy);
        }


        [TestMethod]
        public void TestSorterDistr2()
        {
            const int order = 5;
            const int stageCount = 2;
            const int sorterCount = 100000;

            var randy = Rando.Standard(1444);
            var p1 = randy.RandomSorterPool(order, stageCount, sorterCount);

            var distr = p1.ToSorterDistr();

        }
    }
}
