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
            var res = FloatArrayGen.LeftRightGradient(8, 0.0f, 1.0f);
        }

    }
}
