using System;
using System.Diagnostics;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Utils;

namespace SpongeTester
{
    [TestClass]
    public class GaDualSorterFixture
    {
        [TestMethod]
        public void TestMakeGaDualSorter()
        {
            const int seed = 5234;
            const int order = 17;
            const int stageCount = 20;
            const int genomeCount = 20;
            const int sortableCount = 10;

            var randy = Rando.Standard(seed);
            var genomePoolDualSorter = randy.ToGenomePoolDualSorter(order, stageCount, genomeCount);

            var randomSortablePool = randy.RandomSortablePool(
                order: order,
                poolCount: sortableCount);

            var gaDualSorter = new GaDualSorter(
                genomePoolDualSorter: genomePoolDualSorter,
                sortablePool: randomSortablePool,
                randy:randy);
        }

        [TestMethod]
        public void TestRefineGa()
        {
            const int seed = 7234;
            const int order = 36;
            const int stageCount = 12;
            const int genomeCount = 256;
            const int sortableCount = 128;
            const int selectionFactor = 16;
            const int rounds = 7500;
            const int srounds = 4;

            var sw = new Stopwatch();
            sw.Start();

            var randy = Rando.Standard(seed);
            var genomePoolDualSorter = randy.ToGenomePoolDualSorter(order, stageCount, genomeCount);

            var randomSortablePool = randy.RandomSortablePool(
                order: order,
                poolCount: sortableCount);

            var gaDualSorter = new GaDualSorter(
                genomePoolDualSorter: genomePoolDualSorter,
                sortablePool: randomSortablePool,
                randy: randy);

            var gaDualSorterR = new GaDualSorter(
                genomePoolDualSorter: genomePoolDualSorter,
                sortablePool: randomSortablePool,
                randy: randy);


            Console.WriteLine("n_Avg n_Min r_Avg r_Min");

            for (var i = 0; i < rounds; i++)
            {
                GaGenomeDualSorterResult eval = null;
                GaGenomeDualSorterResult evalR = null;
                for (var j = 0; j < srounds; j++)
                {
                    eval = gaDualSorter.Eval(false);
                    evalR = gaDualSorterR.Eval(false);
                    gaDualSorter = eval.EvolveSorters(randy, selectionFactor);
                    gaDualSorterR = eval.EvolveSortersRecomb(randy, selectionFactor);
                }

                var avgE = eval.SorterResults.Select(sr => sr.Value.AverageSortedness).ToList();
                var avgR = evalR.SorterResults.Select(sr => sr.Value.AverageSortedness).ToList();

                Console.WriteLine($"{avgE.Average()} {avgE.Min()} {avgR.Average()} {avgR.Min()}");
            }

            sw.Stop();
            Console.WriteLine("\nElapsed={0}", sw.Elapsed);
        }

    }
}
