using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Utils;

namespace SpongeTester
{
    [TestClass]
    public class BitsFixture
    {

        [TestMethod]
        public void TestBitly()
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
            var ints = tB.ToIntArray();
            var hc = tB.HotCount();

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
            byte[] b_b = { (byte)~a_i, (byte)~b_i };


            var res = a_b.BitOverlap(b_b);
        }


        [TestMethod]
        public void TestTwoCycleCount()
        {
            for (var i = 0; i < 16; i++)
            {
                Console.WriteLine($"{i}  {IntFuncs.Factorial(i)}  {IntFuncs.TwoCycleCount(i)}");
            }
        }



    }
}
