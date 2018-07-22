using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace Utils.Test
{
    [TestClass]
    public class UnitTest1
    {
        [TestMethod]
        public void TestMethod1()
        {
            var grid = new int[] { 0, 0, 0, 0, 1, 0, 0, 0, 0 };

            var res = GridFuncs.Energy4(grid, 3);
        }
    }
}
