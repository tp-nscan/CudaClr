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
    }
}
