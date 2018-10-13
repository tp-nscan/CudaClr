using System;
using System.Diagnostics;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Utils;
using Utils.Ga;
using Utils.Ga.Parts;
using Utils.Sorter;

namespace SpongeTester
{
    [TestClass]
    public class GaDirectSortingFixture
    {
        [TestMethod]
        public void TestRecomboSortingGaData()
        {
            const int seed = 23196;
            const uint order = 256;
            const int stageCount = 20;
            const int sorterCount = 128;
            const int sortableCount = 128;
            const double sorterWinRate = 0.25;
            const double sortableWinRate = 0.25;
            const StageReplacementMode stageReplacementMode = StageReplacementMode.RandomRewire;
            const int batchSize = 10;
            const int batchRounds = 10;

            var randy = Rando.Standard(seed);

            var dsg = randy.ToDirectGaSortingData(
                order: order,
                sorterCount: sorterCount,
                sortableCount: sortableCount,
                stageCount:stageCount,
                sorterWinRate:sorterWinRate,
                sortableWinRate:sortableWinRate,
                stageReplacementMode:stageReplacementMode
            );

            var dsg2 = randy.ToDirectGaSortingData(
                order: order,
                sorterCount: sorterCount,
                sortableCount: sortableCount,
                stageCount: stageCount,
                sorterWinRate: sorterWinRate,
                sortableWinRate: sortableWinRate,
                stageReplacementMode: stageReplacementMode
            );

            Stopwatch sw = new Stopwatch();
            sw.Start();

            for (var j = 0; j < batchRounds; j++)
            {
                for (var i = 0; i < batchSize; i++)
                {
                    dsg = dsg.EvolveSortersConjSortablesReplace(randy);
                    //dsg2 = dsg2.EvolveBothAndRecombineDirect(randy);
                    Console.WriteLine($"{dsg.Report()}"); // { dsg2.Report()}");
                }

            }

            sw.Stop();
            Console.WriteLine("\nElapsed={0}", sw.Elapsed);
        }


        [TestMethod]
        public void TestCompareSortingGaData()
        {
            const int seed = 23196;
            const uint order = 48;
            const int stageCount = 18;
            const int sorterCount = 256;
            const int sortableCount = 256;
            const double sorterWinRate = 0.25;
            const double sortableWinRate = 0.75;
            const StageReplacementMode stageReplacementMode = StageReplacementMode.RandomRewire;
            const int rounds = 200;
            const uint rollingListCap = 10;

            var randy = Rando.Standard(seed);

            var dsgOld = randy.ToDirectGaSortingData(
                order: order,
                sorterCount: sorterCount,
                sortableCount: sortableCount,
                stageCount: stageCount,
                sorterWinRate: sorterWinRate,
                sortableWinRate: sortableWinRate,
                stageReplacementMode: stageReplacementMode
            );

            var rlSgd = new RollingList<GaSortingData>(rollingListCap);

            for (var i = 0; i < rollingListCap; i++)
            {
                var dsgNew = dsgOld.EvolveSortersDirect(randy);
                rlSgd.Add(dsgNew);
                dsgOld = dsgNew;
            }

           
            Stopwatch sw = new Stopwatch();
            sw.Start();
            Console.Write($"\n");

            for (var i = 0; i < rounds; i++)
            {
                var dsgNew = dsgOld.EvolveSortersConjSortablesReplace(randy);
                var rl = rlSgd.Select(sgd => sgd.CompareReport(dsgNew)).ToList();
                foreach (var r in rl)
                {
                    Console.Write($"{r} ");
                }
                Console.Write($"\n");

                dsgOld = dsgNew;
                rlSgd.Add(dsgNew);
            }


            sw.Stop();
            Console.WriteLine("\nElapsed={0}", sw.Elapsed);
        }


    }
}
