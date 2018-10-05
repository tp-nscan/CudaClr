using System;
using System.Diagnostics;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Utils;
using Utils.Ga;
using Utils.Genome.Utils;
using Utils.Sorter;

namespace SpongeTester
{
    [TestClass]
    public class GaDimerFixture
    {
        [TestMethod]
        public void TestToRandomStageDimerGaData()
        {
            const int seed = 23196;
            const uint order = 48;
            const int stageCount = 18;
            const int sorterCount = 64;
            const int sortableCount = 64;
            const double sorterWinRate = 0.25;
            const double sortableWinRate = 0.25;
            const StageReplacementMode stageReplacementMode = StageReplacementMode.RandomRewire;
            const int batchSize = 10;
            const int batchRounds = 20;

            var randy = Rando.Standard(seed);

            var ds = randy.ToRandomStageDimerGaData(
                    order: order,
                    sorterCount: sorterCount,
                    sortableCount: sortableCount,
                    stageCount: stageCount,
                    sorterWinRate: sorterWinRate,
                    sortableWinRate: sortableWinRate
                );

            var dsg = randy.ToRandomDirectSortingGaData(
                order: order,
                sorterCount: sorterCount,
                sortableCount: sortableCount,
                stageCount: stageCount,
                sorterWinRate: sorterWinRate,
                sortableWinRate: sortableWinRate,
                stageReplacementMode: stageReplacementMode
            );


            var sw = new Stopwatch();
            sw.Start();

            for (var j = 0; j < batchRounds; j++)
            {
                for (var i = 0; i < batchSize * 2; i++)
                {
                    dsg = dsg.EvolveBothDirect(randy);
                    ds = ds.EvolveStageDimerSorters(randy);
                    Console.WriteLine($"{ds.Report()} {dsg.Report()}");
                }
                
            }

            sw.Stop();
            Console.WriteLine("\nElapsed={0}", sw.Elapsed);

        }
    }
}
