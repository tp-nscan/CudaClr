using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Utils;

namespace SpongeTester
{
    [TestClass]
    public class BitsFixture
    {

        [TestMethod]
        public void TestOneHotInts()
        {
            var res = Bitly.OneHotInts;
        }

        [TestMethod]
        public void TestMutateAbit()
        {
            var randy = Rando.Standard(1444);
            uint start = 213533;
            var mutato = start.MutateAbit(randy);
            var startArray = start.ToIntArray();
            var mutArray = mutato.ToIntArray();
            var diff = start ^ mutato;
            var diffArray = diff.ToIntArray();
        }

        [TestMethod]
        public void TestTryThis()
        {
            const uint one = 1;
            const uint two = 2;
            const uint three = 3;
            const uint max = UInt32.MaxValue;

            var res = Bitly.ToIntArray(one);
            res = Bitly.ToIntArray(two);
            res = Bitly.ToIntArray(three);
            res = Bitly.ToIntArray(max);

            var restt = Bitly.TryThis();
        }


        [TestMethod]
        public void TestHotCount()
        {
            const byte tB = 211;
            var bints = tB.ToIntArray();
            var hcb = tB.HotCount();

            const uint tU = 1442342;
            var uints = tU.ToIntArray();
            var hcu = tU.HotCount();
        }

        [TestMethod]
        public void TestByteOverlap()
        {
            int a_i = 1 + 4 + 16 + 64 + 128;
            int b_i = 1 + 2 + 16 + 128;

            byte a_b = (byte)a_i;
            byte b_b = (byte)b_i;


            var res = a_b.BitOverlap(b_b);
        }


        [TestMethod]
        public void TestByteArrayOverlap()
        {
            int a_i = 1 + 4 + 16 + 64 + 128;
            int b_i = 1 + 2 + 16 + 128;

            byte[] a_b = { (byte)a_i, (byte)b_i };
            byte[] b_b = { (byte)a_i, (byte)b_i };


            var res = a_b.BitOverlaps(b_b);
        }


        [TestMethod]
        public void TestTwoCycleCount()
        {
            for (var i = 0; i < 16; i++)
            {
                Console.WriteLine($"{i}  {IntFuncs.Factorial(i)}  {IntFuncs.TwoCycleCount(i)}");
            }
        }

        [TestMethod]
        public void TestNextUint()
        {
            IRando randy = Rando.Standard(123);
            for (var i = 0; i < 16; i++)
            {
                Console.WriteLine($"{randy.NextUint()}");
            }
        }

    }
}
