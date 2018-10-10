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
        public void TestToRandomStageDimerGaData128()
        {
            const uint order = 128;
            const int stageCount = 18;
            const int sorterCount = 256;
            const int sortableCount = 32;
            const double sorterWinRate = 0.25;
            const double sortableWinRate = 0.25;
            const StageReplacementMode stageReplacementMode = StageReplacementMode.RandomRewire;
            const int batchSize = 10;
            const int batchRounds = 2;

            const int seed = 75319;
            var randy = Rando.Standard(seed);

            var dga1 = randy.ToRandomDirectSortingGaData(
                order: order,
                sorterCount: sorterCount,
                sortableCount: sortableCount,
                stageCount: stageCount,
                sorterWinRate: sorterWinRate,
                sortableWinRate: sortableWinRate,
                stageReplacementMode:stageReplacementMode
            );

            var dga2 = randy.ToRandomDirectSortingGaData(
                order: order,
                sorterCount: sorterCount,
                sortableCount: sortableCount,
                stageCount: stageCount,
                sorterWinRate: sorterWinRate,
                sortableWinRate: sortableWinRate,
                stageReplacementMode: stageReplacementMode
            );
            var dga3 = randy.ToRandomDirectSortingGaData(
                order: order,
                sorterCount: sorterCount,
                sortableCount: sortableCount,
                stageCount: stageCount,
                sorterWinRate: sorterWinRate,
                sortableWinRate: sortableWinRate,
                stageReplacementMode: stageReplacementMode
            );
            var dga4 = randy.ToRandomDirectSortingGaData(
                order: order,
                sorterCount: sorterCount,
                sortableCount: sortableCount,
                stageCount: stageCount,
                sorterWinRate: sorterWinRate,
                sortableWinRate: sortableWinRate,
                stageReplacementMode: stageReplacementMode
            );



            var sdgca1 = randy.ToRandomStageDimerGaData(
                    order: order,
                    sorterCount: sorterCount,
                    sortableCount: sortableCount,
                    stageCount: stageCount,
                    sorterWinRate: sorterWinRate,
                    sortableWinRate: sortableWinRate
                );

            var sdgca2 = randy.ToRandomStageDimerGaData(
                order: order,
                sorterCount: sorterCount,
                sortableCount: sortableCount,
                stageCount: stageCount,
                sorterWinRate: sorterWinRate,
                sortableWinRate: sortableWinRate
            );

            var sdgca3 = randy.ToRandomStageDimerGaData(
                order: order,
                sorterCount: sorterCount,
                sortableCount: sortableCount,
                stageCount: stageCount,
                sorterWinRate: sorterWinRate,
                sortableWinRate: sortableWinRate
            );

            var sdgca4 = randy.ToRandomStageDimerGaData(
                order: order,
                sorterCount: sorterCount,
                sortableCount: sortableCount,
                stageCount: stageCount,
                sorterWinRate: sorterWinRate,
                sortableWinRate: sortableWinRate
            );

            var sw = new Stopwatch();
            sw.Start();

            for (var j = 0; j < batchRounds; j++)
            {
                for (var i = 0; i < batchSize * 2; i++)
                {
                    dga1 = dga1.EvolveBothAndRecombineDirect(randy);
                    dga2 = dga2.EvolveBothAndRecombineDirect(randy);
                    dga3 = dga3.EvolveBothAndRecombineDirect(randy);
                    dga4 = dga4.EvolveBothAndRecombineDirect(randy);


                    sdgca1 = sdgca1.EvolveRecombineCoarseStageDimerSortersAndSortables(randy);
                    sdgca2 = sdgca2.EvolveRecombineCoarseStageDimerSortersAndSortables(randy);
                    sdgca3 = sdgca3.EvolveRecombineFineStageDimerSortersAndSortables(randy);
                    sdgca4 = sdgca4.EvolveRecombineFineStageDimerSortersAndSortables(randy);

                    Console.WriteLine($"{dga1.Report()} {dga2.Report()} " +
                                      $"{dga3.Report()} {dga4.Report()} " +
                                      $"{sdgca1.Report()} {sdgca2.Report()} " +
                                      $"{sdgca3.Report()} {sdgca4.Report()} ");
                }
            }

            sw.Stop();
            Console.WriteLine("\nElapsed={0}", sw.Elapsed);

        }


        [TestMethod]
        public void TestToRandomStageDimerGaDataVaryPoolSizes()
        {
            const uint order = 128;
            const int stageCount = 18;
            const int sorterCount = 256;
            const int sortableCount = 32;
            const double sorterWinRate = 0.25;
            const double sortableWinRate = 0.25;
            const StageReplacementMode stageReplacementMode = StageReplacementMode.RandomRewire;
            const int batchSize = 10;
            const int batchRounds = 200;

            const int seed = 75333;
            var randy = Rando.Standard(seed);


            var sdgca1 = randy.ToRandomStageDimerGaData(
                    order: order,
                    sorterCount: sorterCount,
                    sortableCount: sortableCount,
                    stageCount: stageCount,
                    sorterWinRate: sorterWinRate,
                    sortableWinRate: sortableWinRate
                );

            var sdgca2 = randy.ToRandomStageDimerGaData(
                order: order,
                sorterCount: sorterCount,
                sortableCount: sortableCount * 2,
                stageCount: stageCount,
                sorterWinRate: sorterWinRate,
                sortableWinRate: sortableWinRate
            );

            var sdgca3 = randy.ToRandomStageDimerGaData(
                order: order,
                sorterCount: sorterCount,
                sortableCount: sortableCount * 4,
                stageCount: stageCount,
                sorterWinRate: sorterWinRate,
                sortableWinRate: sortableWinRate
            );

            var sdgca4 = randy.ToRandomStageDimerGaData(
                order: order,
                sorterCount: sorterCount / 2,
                sortableCount: sortableCount * 4,
                stageCount: stageCount,
                sorterWinRate: sorterWinRate,
                sortableWinRate: sortableWinRate
            );

            var sw = new Stopwatch();
            sw.Start();

            for (var j = 0; j < batchRounds; j++)
            {
                for (var i = 0; i < batchSize * 2; i++)
                {
                    sdgca1 = sdgca1.EvolveRecombineCoarseStageDimerSortersAndSortables(randy);
                    sdgca2 = sdgca2.EvolveRecombineCoarseStageDimerSortersAndSortables(randy);
                    sdgca3 = sdgca3.EvolveRecombineCoarseStageDimerSortersAndSortables(randy);
                    sdgca4 = sdgca4.EvolveRecombineCoarseStageDimerSortersAndSortables(randy);

                    Console.WriteLine(
                                      $"{sdgca1.Report()} {sdgca2.Report()} " +
                                      $"{sdgca3.Report()} {sdgca4.Report()} ");
                }
            }

            sw.Stop();
            Console.WriteLine("\nElapsed={0}", sw.Elapsed);

        }




        [TestMethod]
        public void TestToRandomStageDimerGaData64()
        {
            const uint order = 64;
            const int stageCount = 12;
            const int sorterCount = 128;
            const int sortableCount = 32;
            const double sorterWinRate = 0.25;
            const double sortableWinRate = 0.25;
            const StageReplacementMode stageReplacementMode = StageReplacementMode.RandomRewire;
            const int batchSize = 10;
            const int batchRounds = 50;

            const int seed = 75319;
            var randy = Rando.Standard(seed);

            var dga1 = randy.ToRandomDirectSortingGaData(
                order: order,
                sorterCount: sorterCount,
                sortableCount: sortableCount,
                stageCount: stageCount,
                sorterWinRate: sorterWinRate,
                sortableWinRate: sortableWinRate,
                stageReplacementMode: stageReplacementMode
            );

            var dga2 = randy.ToRandomDirectSortingGaData(
                order: order,
                sorterCount: sorterCount,
                sortableCount: sortableCount,
                stageCount: stageCount,
                sorterWinRate: sorterWinRate,
                sortableWinRate: sortableWinRate,
                stageReplacementMode: stageReplacementMode
            );
            var dga3 = randy.ToRandomDirectSortingGaData(
                order: order,
                sorterCount: sorterCount,
                sortableCount: sortableCount,
                stageCount: stageCount,
                sorterWinRate: sorterWinRate,
                sortableWinRate: sortableWinRate,
                stageReplacementMode: stageReplacementMode
            );
            var dga4 = randy.ToRandomDirectSortingGaData(
                order: order,
                sorterCount: sorterCount,
                sortableCount: sortableCount,
                stageCount: stageCount,
                sorterWinRate: sorterWinRate,
                sortableWinRate: sortableWinRate,
                stageReplacementMode: stageReplacementMode
            );



            var sdgca1 = randy.ToRandomStageDimerGaData(
                    order: order,
                    sorterCount: sorterCount,
                    sortableCount: sortableCount,
                    stageCount: stageCount,
                    sorterWinRate: sorterWinRate,
                    sortableWinRate: sortableWinRate
                );

            var sdgca2 = randy.ToRandomStageDimerGaData(
                order: order,
                sorterCount: sorterCount,
                sortableCount: sortableCount,
                stageCount: stageCount,
                sorterWinRate: sorterWinRate,
                sortableWinRate: sortableWinRate
            );

            var sdgca3 = randy.ToRandomStageDimerGaData(
                order: order,
                sorterCount: sorterCount,
                sortableCount: sortableCount,
                stageCount: stageCount,
                sorterWinRate: sorterWinRate,
                sortableWinRate: sortableWinRate
            );

            var sdgca4 = randy.ToRandomStageDimerGaData(
                order: order,
                sorterCount: sorterCount,
                sortableCount: sortableCount,
                stageCount: stageCount,
                sorterWinRate: sorterWinRate,
                sortableWinRate: sortableWinRate
            );

            var sw = new Stopwatch();
            sw.Start();

            for (var j = 0; j < batchRounds; j++)
            {
                for (var i = 0; i < batchSize * 2; i++)
                {
                    dga1 = dga1.EvolveBothAndRecombineDirect(randy);
                    dga2 = dga2.EvolveBothAndRecombineDirect(randy);
                    dga3 = dga3.EvolveBothAndRecombineDirect(randy);
                    dga4 = dga4.EvolveBothAndRecombineDirect(randy);


                    sdgca1 = sdgca1.EvolveRecombineCoarseStageDimerSortersAndSortables(randy);
                    sdgca2 = sdgca2.EvolveRecombineCoarseStageDimerSortersAndSortables(randy);
                    sdgca3 = sdgca3.EvolveRecombineFineStageDimerSortersAndSortables(randy);
                    sdgca4 = sdgca4.EvolveRecombineFineStageDimerSortersAndSortables(randy);

                    Console.WriteLine($"{dga1.Report()} {dga2.Report()} " +
                                      $"{dga3.Report()} {dga4.Report()} " +
                                      $"{sdgca1.Report()} {sdgca2.Report()} " +
                                      $"{sdgca3.Report()} {sdgca4.Report()} ");
                }
            }

            sw.Stop();
            Console.WriteLine("\nElapsed={0}", sw.Elapsed);

        }


        [TestMethod]
        public void TestToRandomStageDimerGaData256()
        {
            const uint order = 256;
            const int stageCount = 24;
            const int sorterCount = 512;
            const int sortableCount = 8;
            const double sorterWinRate = 0.25;
            const double sortableWinRate = 0.25;
            const StageReplacementMode stageReplacementMode = StageReplacementMode.RandomRewire;
            const int batchSize = 5;
            const int batchRounds = 2;

            const int seed = 75319;
            var randy = Rando.Standard(seed);

            var dga1 = randy.ToRandomDirectSortingGaData(
                order: order,
                sorterCount: sorterCount,
                sortableCount: sortableCount,
                stageCount: stageCount,
                sorterWinRate: sorterWinRate,
                sortableWinRate: sortableWinRate,
                stageReplacementMode: stageReplacementMode
            );

            var dga2 = randy.ToRandomDirectSortingGaData(
                order: order,
                sorterCount: sorterCount,
                sortableCount: sortableCount,
                stageCount: stageCount,
                sorterWinRate: sorterWinRate,
                sortableWinRate: sortableWinRate,
                stageReplacementMode: stageReplacementMode
            );
            var dga3 = randy.ToRandomDirectSortingGaData(
                order: order,
                sorterCount: sorterCount,
                sortableCount: sortableCount,
                stageCount: stageCount,
                sorterWinRate: sorterWinRate,
                sortableWinRate: sortableWinRate,
                stageReplacementMode: stageReplacementMode
            );
            var dga4 = randy.ToRandomDirectSortingGaData(
                order: order,
                sorterCount: sorterCount,
                sortableCount: sortableCount,
                stageCount: stageCount,
                sorterWinRate: sorterWinRate,
                sortableWinRate: sortableWinRate,
                stageReplacementMode: stageReplacementMode
            );



            var sdgca1 = randy.ToRandomStageDimerGaData(
                    order: order,
                    sorterCount: sorterCount,
                    sortableCount: sortableCount,
                    stageCount: stageCount,
                    sorterWinRate: sorterWinRate,
                    sortableWinRate: sortableWinRate
                );

            var sdgca2 = randy.ToRandomStageDimerGaData(
                order: order,
                sorterCount: sorterCount,
                sortableCount: sortableCount,
                stageCount: stageCount,
                sorterWinRate: sorterWinRate,
                sortableWinRate: sortableWinRate
            );

            var sdgca3 = randy.ToRandomStageDimerGaData(
                order: order,
                sorterCount: sorterCount,
                sortableCount: sortableCount,
                stageCount: stageCount,
                sorterWinRate: sorterWinRate,
                sortableWinRate: sortableWinRate
            );

            var sdgca4 = randy.ToRandomStageDimerGaData(
                order: order,
                sorterCount: sorterCount,
                sortableCount: sortableCount,
                stageCount: stageCount,
                sorterWinRate: sorterWinRate,
                sortableWinRate: sortableWinRate
            );

            var sw = new Stopwatch();
            sw.Start();

            for (var j = 0; j < batchRounds; j++)
            {
                for (var i = 0; i < batchSize; i++)
                {
                    dga1 = dga1.EvolveBothAndRecombineDirect(randy);
                    dga2 = dga2.EvolveBothAndRecombineDirect(randy);
                    dga3 = dga3.EvolveBothAndRecombineDirect(randy);
                    dga4 = dga4.EvolveBothAndRecombineDirect(randy);


                    sdgca1 = sdgca1.EvolveRecombineCoarseStageDimerSortersAndSortables(randy);
                    sdgca2 = sdgca2.EvolveRecombineCoarseStageDimerSortersAndSortables(randy);
                    sdgca3 = sdgca3.EvolveRecombineCoarseStageDimerSortersAndSortables(randy);
                    sdgca4 = sdgca4.EvolveRecombineCoarseStageDimerSortersAndSortables(randy);

                    Console.WriteLine($"{dga1.Report()} {dga2.Report()} " +
                                      $"{dga3.Report()} {dga4.Report()} " +
                                      $"{sdgca1.Report()} {sdgca2.Report()} " +
                                      $"{sdgca3.Report()} {sdgca4.Report()} ");
                }
            }

            sw.Stop();
            Console.WriteLine("\nElapsed={0}", sw.Elapsed);

        }

    }
}
