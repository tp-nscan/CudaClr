using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Utils;

namespace SpongeTester
{
    [TestClass]
    public class UnitTest1
    {
        public const int Order = 10;

        [TestMethod]
        public void TestMethod1()
        {
            Console.WriteLine("hi");
        }


        [TestMethod]
        public void TestRandomTwoCycleSquared()
        {
            var rando = Rando.Standard(5123);
            var perm = rando.RandomPermutation(Order);

            var p2 = perm.Multiply(perm);

            Assert.IsTrue(p2.IsEqualTo(PermutationEx.Identity(Order)));
        }


        [TestMethod]
        public void TestRandomTwoCycle()
        {
            var randy = Rando.Standard(5123);

            for (var i = 0; i < 1000; i++)
            {
                var perm = randy.RandomSorterStage(Order);
                var p2 = perm. Multiply(perm);
                Assert.IsTrue(p2.IsEqualTo(PermutationEx.Identity(Order)));
            }
        }


        [TestMethod]
        public void TestHc()
        {
            var pd = PermutationEx.MakeCompositeDictionaryForPermutation(Order);

            var randy = Rando.Standard(51323);

            for (var i = 0; i < 200000; i++)
            {
                var perm = randy.RandomPermutation(Order);
                if (pd.ContainsKey(perm))
                {
                    pd[perm]++;
                }
                else
                {
                    pd.Add(perm, 1);
                }
            }
        }


        [TestMethod]
        public void TestOrbs()
        {
            const int trials = 100000;
            List<int> res = new List<int>();

            var randy = Rando.Standard(1444);
            
            for (var i = 0; i < trials; i++)
            {
                var perm = randy.RandomPermutation(Order);
                res.Add(OrbitLengthFor(perm));
            }

            var grouped = res.GroupBy(i => i);

            foreach (var g in grouped.OrderBy(gg=>gg.Key))
            {
                Console.WriteLine($"{g.Key} {g.Count()}");
            }
        }


        int OrbitLengthFor(IPermutation perm)
        {
            var pd = PermutationEx.MakeCompositeDictionaryForPermutation(Order);
            var cume = perm;
            for (var i = 0; i < 10000; i++)
            {
                if (pd.ContainsKey(cume))
                {
                    return pd.Count;
                }
                else
                {
                    pd.Add(cume, 1);
                }
                cume = cume.Multiply(perm);

            }
            return -1;
        }


        [TestMethod]
        public void TestOrbsAgainstOrderliness()
        {
            const int trials = 1000;

            var randy = Rando.Standard(1444);

            for (var i = 0; i < trials; i++)
            {
                var perm = randy.RandomPermutation(Order);
                Console.WriteLine($"{OrbitLengthFor(perm)} {perm.OutOfOrderliness()}");
            }
        }


        [TestMethod]
        public void TestSortliness()
        {
            const int trials = 5000;
            var randy = Rando.Standard(1444);

            for (var i = 0; i < trials; i++)
            {
                var stagey = randy.RandomSorterStage(Order);
                var stagey2 = randy.RandomSorterStage(Order);
                var stagey3 = randy.RandomSorterStage(Order);
                var stagey4 = randy.RandomSorterStage(Order);

                var permy = randy.RandomPermutation(Order);

                var strey = stagey.Sort(permy);
                var strey2 = stagey2.Sort(strey.Item2);
                var strey3 = stagey3.Sort(strey2.Item2);
                var strey4 = stagey4.Sort(strey3.Item2);
                var strey5 = stagey.Sort(strey4.Item2);
                var strey6 = stagey2.Sort(strey5.Item2);
                var strey7 = stagey3.Sort(strey6.Item2);
                var strey8 = stagey4.Sort(strey7.Item2);

                var pr = permy.OutOfOrderliness();
                var psr = strey.Item2.OutOfOrderliness();
                var psr2 = strey8.Item2.OutOfOrderliness();

                Console.WriteLine($"{pr} {psr} {psr2}");
            }
        }

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
        public void TestByteArrayMatch()
        {
            int a_i = 1 +      4 + 16 + 64 + 128;
            int b_i = 1 + 2 +      16      + 128;

            byte a_b = (byte) a_i;
            byte b_b = (byte) b_i;


            var res = a_b.BitOverlap(b_b);
        }

    }
}
