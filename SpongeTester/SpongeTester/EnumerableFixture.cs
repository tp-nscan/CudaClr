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
    }
}
