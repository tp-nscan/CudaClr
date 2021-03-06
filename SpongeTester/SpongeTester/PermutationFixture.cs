﻿using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Utils;
using Utils.Sorter;

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
            var perm = rando.ToPermutation(order);

            var p2 = perm.Multiply(perm);

            Assert.IsTrue(p2.IsEqualTo(PermutationEx.Identity(order)));
        }


        [TestMethod]
        public void TestInverse()
        {
            const int order = 23;
            var randy = Rando.Standard(5123);
            for (var i = 0; i < 1000; i++)
            {
                var perm = randy.ToPermutation(order);
                var permI = perm.ToInverse();
                var prod = perm.Multiply(permI);
                Assert.IsTrue(prod.IsEqualTo(PermutationEx.Identity(order)));
            }
        }

        [TestMethod]
        public void TestRandomTwoCycleStage()
        {
            const int order = 24;
            var randy = Rando.Standard(5123);
            for (var i = 0; i < 1000; i++)
            {
                var twoCycle = randy.ToFullTwoCyclePermutation(order);
                var twoCycleSq = twoCycle.Multiply(twoCycle);
                Assert.IsTrue(twoCycleSq.IsEqualTo(PermutationEx.Identity(order)));
            }
        }


        [TestMethod]
        public void TestFullTwoCyclePermutationConjugateIsATwoCycle()
        {
            const int order = 24;
            var randy = Rando.Standard(5123);
            for (var i = 0; i < 1000; i++)
            {
                var twoCycle = randy.ToFullTwoCyclePermutation(order);
                var conjugate = twoCycle.ConjugateByRandomPermutation(randy);
                var p2 = conjugate.Multiply(conjugate);
                Assert.IsTrue(p2.IsEqualTo(PermutationEx.Identity(order)));
            }
        }

        [TestMethod]
        public void TestSingleTwoCyclePermutationConjugateIsATwoCycle()
        {
            const int order = 24;
            var randy = Rando.Standard(5123);
            for (var i = 0; i < 1000; i++)
            {
                var twoCycle = randy.ToSingleTwoCyclePermutation(order);
                var twoCycleSq = twoCycle.Multiply(twoCycle);
                Assert.IsTrue(twoCycleSq.IsEqualTo(PermutationEx.Identity(order)));
            }
        }


        [TestMethod]
        public void TestConjugateByRandomSingleTwoCycle()
        {
            const int order = 24;
            var randy = Rando.Standard(5123);
            var perm = randy.ToPermutation(order);

            for (var i = 0; i < 1000; i++)
            {
                perm = perm.ConjugateByRandomSingleTwoCycle(randy);
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
                var perm = randy.ToFullSorterStage(orderly, 0);
                var perm2 = randy.RewireSorterStage(perm);

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
        public void TestPermutationOrbs()
        {
            const int order = 128;
            const int trials = 50;
            List<int> res = new List<int>();

            var randy = Rando.Standard(3754);

            for (var i = 0; i < trials; i++)
            {
                var perm = randy.ToPermutation(order);
                res.Add(perm.OrbitLengthFor(10000));
            }

            var grouped = res.GroupBy(i => i);

            foreach (var g in grouped.OrderBy(gg => gg.Key))
            {
                Console.WriteLine($"{g.Key} {g.Count()}");
            }
        }

        [TestMethod]
        public void TestTwoCycleConjPermutationOrbs()
        {
            const int order = 128;
            const int trials = 50;
            List<int> res = new List<int>();

            var randy = Rando.Standard(7374);
            var fullTwoCyclePermutation = randy.ToFullTwoCyclePermutation(order);

            for (var i = 0; i < trials; i++)
            {
                var perm = randy.ToPermutation(order);

                res.Add(fullTwoCyclePermutation.ConjOrbitLengthFor(perm, 10000));
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
                var perm = randy.ToPermutation(order);
                Console.WriteLine($"{perm.OrbitLengthFor()} {perm.SortednessSq()}");
            }
        }

        [TestMethod]
        public void TestToRandomPairs()
        {
            const int order = 24;
            var randy = Rando.Standard(1444);

            var src = Enumerable.Range(0, order).ToRandomPairs(randy).ToList();

        }

    }
}
