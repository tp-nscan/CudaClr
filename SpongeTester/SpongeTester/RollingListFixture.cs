using System;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Utils;

namespace SpongeTester
{
    [TestClass]
    public class RollingListFixture
    {
        [TestMethod]
        public void TestRollingList()
        {
            var rl = new RollingList<uint>(10);

            for (uint i = 0; i < 20; i++)
            {
                rl.Add(i);
            }

            var res = rl.ToList();
        }
    }
}
