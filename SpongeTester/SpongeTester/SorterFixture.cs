using System;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Utils;
using Utils.Sortable;
using Utils.Sorter;

namespace SpongeTester
{
    [TestClass]
    public class SorterFixture
    {

        [TestMethod]
        public void TestSortliness()
        {
            var seed = 1223;
            const int order = 24;
            const int trials = 5000;
            var randy = Rando.Standard(seed);

            for (var i = 0; i < trials; i++)
            {
                var stagey = randy.ToFullSorterStage(order, 0);
                var stagey2 = randy.ToFullSorterStage(order, 0);
                var stagey3 = randy.ToFullSorterStage(order, 0);
                var stagey4 = randy.ToFullSorterStage(order, 0);

                var permy = randy.ToPermutation(order);

                var strey0 = stagey.Sort(permy);
                var strey2 = stagey2.Sort(strey0.Item2);
                var strey3 = stagey3.Sort(strey2.Item2);
                var strey4 = stagey4.Sort(strey3.Item2);
                var strey5 = stagey.Sort(strey4.Item2);
                var strey6 = stagey2.Sort(strey5.Item2);
                var strey7 = stagey3.Sort(strey6.Item2);
                var strey8 = stagey4.Sort(strey7.Item2);

                var pr = permy.SortednessSq();
                var psr0 = strey0.Item2.SortednessSq();
                var psr2 = strey8.Item2.SortednessSq();

                Console.WriteLine($"{pr} {psr0} {psr2}");
            }
        }


        [TestMethod]
        public void TestSorterSort()
        {
            var seed = 1223;
            const uint order = 24;
            var permCount = 2000u;
            var sorterCount = 200u;
            var randy = Rando.Standard(seed);

            var permies = 0u.CountUp(permCount)
                .Select(i => randy.ToPermutation(order).ToSortable())
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
            var seed = 1223;
            const uint order = 24;
            const uint stageCount = 5;
            var randy = Rando.Standard(seed);
            var oldSorter = randy.ToSorter(order, stageCount);
            var newSorter = oldSorter.Mutate(randy, StageReplacementMode.RandomConjugate);
        }


        [TestMethod]
        public void TestRecombo()
        {
            var seed = 1223;
            const uint order = 24;
            const uint stageCount = 5;
            var randy = Rando.Standard(seed);
            var sorterA = randy.ToSorter(order, stageCount);
            var sorterB = randy.ToSorter(order, stageCount);


        }

    }
}
