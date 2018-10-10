using System;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Utils;

namespace SpongeTester
{
    [TestClass]
    public class EnumerableFixture
    {
        [TestMethod]
        public void TestSquareArrayCoords()
        {
            var res = 4u.SquareArrayCoords().ToList();

        }


        [TestMethod]
        public void TestRoundRobin()
        {
            var wrappy = new[] {0, 1, 2, 4}.ToRoundRobin().Take(17).ToList();
        }


        [TestMethod]
        public void TestRecurse()
        {
            Func<int, int> tf = i => i + 1;
            var res = tf.Recurse(0, 10).ToList();
        }


        [TestMethod]
        public void TestToRandomPairs()
        {
            const int seed = 23196;
            var randy = Rando.Standard(seed);
            var prep = 0u.CountUp(10);

            var posp = prep.ToRandomPairs(randy).ToList();

        }

        [TestMethod]
        public void TestRecomb()
        {
            const int seed = 23196;
            const uint strandLen = 5;
            var randy = Rando.Standard(seed);
            var strandA = 0u.CountUp(strandLen).ToList();
            var strandB = 100u.CountUp(100 + strandLen).ToList();

            for (uint i = 0; i < strandLen + 1; i++)
            {
                var res = strandA.Recombo(strandB, i);
            }
        }

        Tuple<uint, uint> TestCrossPtFunc(uint a, uint b)
        {
            return new Tuple<uint, uint>(b + 1000, a + 10000);
        }

        [TestMethod]
        public void TestRecombL2()
        {
            const int seed = 23196;
            const uint strandLen = 5;
            var randy = Rando.Standard(seed);
            var strandA = 0u.CountUp(strandLen).ToList();
            var strandB = 100u.CountUp(100 + strandLen).ToList();

            for (uint i = 0; i < strandLen + 1; i++)
            {
                var res = strandA.RecomboL2(strandB, i, TestCrossPtFunc);
            }
        }
    }
}
