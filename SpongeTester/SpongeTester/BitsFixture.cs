using System;
using System.Linq;
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
        public void TestBitFlip()
        {
            var randy = Rando.Standard(1444);
            uint start = 213533;
            var mutato = start.BitFlip(2);
            var startArray = start.ToIntArray();
            var mutArray = mutato.ToIntArray();
            var diff = start ^ mutato;
            var diffArray = diff.ToIntArray();
        }


        [TestMethod]
        public void TestTryBitConverter()
        {
            var restt = Bitly.TryBitConverter();
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
            var randy = Rando.Standard(123);
            for (var i = 0; i < 16; i++)
            {
                Console.WriteLine($"{randy.NextUint()}");
            }
        }


        [TestMethod]
        public void TestMutate()
        {
            const uint arrayLen = 1000;
            const double mutationRate = 0.1;
            var randy = Rando.Standard(123);
            var ulongA = 0u.CountUp(arrayLen).Select(i => randy.NextUint()).ToArray();
            var ulongAm = ulongA.Mutate(randy, mutationRate).ToArray();

            var diff = arrayLen * 32 - ulongA.BitOverlaps(ulongAm).Sum(i=>i);
        }

    }
}
