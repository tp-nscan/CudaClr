using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Utils;

namespace SpongeTester
{
    [TestClass]
    public class StageBitsFixture
    {
        [TestMethod]
        public void TestToStageBits()
        {
            const int seed = 5234;
            const uint order = 17;
            const uint mask = 1234;

            var randy = Rando.Standard(seed);
            var stageBits = randy.ToStageBits(order: order, mask:mask);
        }

        [TestMethod]
        public void TestToSbScratchPad()
        {
            const int seed = 5234;
            const uint order = 17;
            const uint mask = 1234;

            var randy = Rando.Standard(seed);
            var stageBits = randy.ToStageBits(order: order, mask: mask);

            var sbScratchPad = stageBits.ToSbScratchPad();
        }

        [TestMethod]
        public void TestToPermutation()
        {
            const int seed = 5234;
            const uint order = 17;
            const uint mask = 1234;

            var randy = Rando.Standard(seed);
            var stageBits = randy.ToStageBits(order: order, mask: mask);

            var perm = stageBits.ToPermutation();
        }
    }
}
