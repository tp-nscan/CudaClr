using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Utils;

namespace SpongeTester
{
    [TestClass]
    public class StringFuncsFixture
    {
        [TestMethod]
        public void TestGridFormat()
        {
            Func<uint, uint, string> f = (r, c) => $"[{r}, {c}]";
            Console.WriteLine(StringFuncs.GridFormat(5, 7, f));
        }
    }
}
