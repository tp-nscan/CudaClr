using System;
using System.Diagnostics;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Utils;
using Utils.Genome;
using Utils.Genome.Utils;

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

            var randy = Rando.Standard(seed);
            var stageBits = randy.ToStageBits(order: order);
        }

        [TestMethod]
        public void TestToSbScratchPad()
        {
            const int seed = 5234;
            const uint order = 17;

            var randy = Rando.Standard(seed);
            var stageBits = randy.ToStageBits(order: order);

            var sbScratchPad = stageBits.ToSbScratchPad();
        }

        [TestMethod]
        public void TestToPermutation()
        {
            const int seed = 5234;
            const uint order = 128;

            var randy = Rando.Standard(seed);

            Stopwatch sw = new Stopwatch();
            sw.Start();

            for (var i = 0; i < 1000; i++)
            {
                var stageBits = randy.ToStageBits(order: order);
                var perm = stageBits.ToPermutation();
                var p2 = perm.Multiply(perm);
              //  Assert.IsTrue(p2.IsEqualTo(PermutationEx.Identity(order)));
            }

            sw.Stop();
            Console.WriteLine("Elapsed={0}", sw.Elapsed);
        }

        [TestMethod]
        public void TestMutate()
        {
            const int seed = 434;
            const uint order = 64;
            const double mutationRate = 0.003;

            var randy = Rando.Standard(seed);
            var stageBits = randy.ToStageBits(order: order);
            var perm = stageBits.ToPermutation();

            Stopwatch sw = new Stopwatch();
            sw.Start();

            var sameCount = 0;
            for (var i = 0; i < 100000; i++)
            {
                var stageBitsM = stageBits.Mutate(randy, mutationRate);
                var permM = stageBitsM.ToPermutation();
                var p2 = perm.Multiply(permM);
                if (p2.IsEqualTo(PermutationEx.Identity(order)))
                {
                    sameCount++;
                }
            }

            sw.Stop();
            Console.WriteLine(sameCount);
            Console.WriteLine("Elapsed={0}", sw.Elapsed);
        }


        [TestMethod]
        public void TestMutatezLoop()
        {
            const int seed = 7434;
            const double mutationRate = 0.001;
            const int reps = 2000;
            const int pool = 2000;

            var randy = Rando.Standard(seed);


            Stopwatch sw = new Stopwatch();
            sw.Start();
            Console.WriteLine($"Order MutationRate P1 P2 Reps");

            for (var p =0; p<10; p++)
            {
                for (uint r = 48; r < 100; r++)
                {
                    var stageBits = randy.ToStageBits(order: r);
                    var perm = stageBits.ToPermutation();

                    var sameCount = 0;
                    var sameCount2 = 0;
                    for (var i = 0; i < reps; i++)
                    {
                        var stageBitsM = stageBits.Mutate(randy, mutationRate);
                        var permM = stageBitsM.ToPermutation();
                        var p2 = perm.Multiply(permM);
                        if (p2.IsEqualTo(PermutationEx.Identity(r)))
                        {
                            sameCount++;
                            var stageBitsM2 = stageBitsM.Mutate(randy, mutationRate);
                            var permM2 = stageBitsM2.ToPermutation();
                            var p3 = perm.Multiply(permM2);
                            if (p3.IsEqualTo(PermutationEx.Identity(r)))
                            {
                                sameCount2++;
                            }
                        }
                    }

                    Console.WriteLine($"{r} {mutationRate} {sameCount} {sameCount2} {reps}");
                }

            }



            sw.Stop();
            Console.WriteLine("\nElapsed={0}", sw.Elapsed);
        }


    }
}
