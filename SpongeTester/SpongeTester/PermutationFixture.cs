using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Utils;

namespace SpongeTester
{
    [TestClass]
    public class PermutationFixture
    {

        [TestMethod]
        public void TestLowerTriangularIndex()
        {
            var r1 = EnumerableExt.LowerTriangularIndex(3, 2);
            var r2 = EnumerableExt.LowerTriangularIndex(2, 2);

        }

        [TestMethod]
        public void TestFromLowerTriangularIndex()
        {
            var r1 = EnumerableExt.FromLowerTriangularIndex(2);
            var r2 = EnumerableExt.FromLowerTriangularIndex(5);
            var r3 = EnumerableExt.FromLowerTriangularIndex(8);
            var r4 = EnumerableExt.FromLowerTriangularIndex(12);
            var r5 = EnumerableExt.FromLowerTriangularIndex(14);
            var r6 = EnumerableExt.FromLowerTriangularIndex(16);
        }

        [TestMethod]
        public void TestRandomTwoCycleSquared()
        {
            const int order = 24;
            var rando = Rando.Standard(5123);
            var perm = rando.RandomPermutation(order);

            var p2 = perm.Multiply(perm);

            Assert.IsTrue(p2.IsEqualTo(PermutationEx.Identity(order)));
        }


        [TestMethod]
        public void TestRandomTwoCycleStage()
        {
            const int order = 24;
            var randy = Rando.Standard(5123);
            for (var i = 0; i < 1000; i++)
            {
                var perm = randy.RandomFullSorterStage(order, 0);
                var p2 = perm.Multiply(perm);
                Assert.IsTrue(p2.IsEqualTo(PermutationEx.Identity(order)));
            }
        }


        [TestMethod]
        public void TestHc()
        {
            uint orderly = 11;
            var pd = PermutationEx.PermutationDictionary(orderly);
            var randy = Rando.Standard(51323);

            for (var i = 0; i < 1000; i++)
            {
                var perm = randy.RandomFullSorterStage(orderly, 0);
                var perm2 = randy.MutateSorterStage(perm);

                if (perm.IsEqualTo(perm2))
                {
                    var s = "s";
                }

                if (pd.ContainsKey(perm2))
                {
                    pd[perm2]++;
                }
                else
                {
                    pd.Add(perm2, 1);
                }
            }
        }


        [TestMethod]
        public void TestOrbs()
        {
            const int order = 16;
            const int trials = 500000;
            List<int> res = new List<int>();

            var randy = Rando.Standard(374);

            for (var i = 0; i < trials; i++)
            {
                var perm = randy.RandomPermutation(order);
                res.Add(perm.OrbitLengthFor(1000000));
            }

            var grouped = res.GroupBy(i => i);

            foreach (var g in grouped.OrderBy(gg => gg.Key))
            {
                Console.WriteLine($"{g.Key} {g.Count()}");
            }
        }


        [TestMethod]
        public void TestOrbsAgainstOrderliness()
        {
            const int order = 24;
            const int trials = 10000;

            var randy = Rando.Standard(1444);

            for (var i = 0; i < trials; i++)
            {
                var perm = randy.RandomPermutation(order);
                Console.WriteLine($"{perm.OrbitLengthFor()} {perm.Sortedness()}");
            }
        }

        [TestMethod]
        public void TestToRandomPairs()
        {
            const int order = 24;
            var randy = Rando.Standard(1444);

            var src = Enumerable.Range(0, order).ToRandomPairs(randy).ToList();

        }

        [TestMethod]
        public void TestRecombine()
        {
            const int order = 24;
            var randy = Rando.Standard(1444);
            var lhs = Enumerable.Range(0, order).ToList();
            var rhs = Enumerable.Range(order, order).ToList();

            var src = randy.Recombo(lhs, rhs);

        }

    }
}
