using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Utils;
using Utils.Ga;
using Utils.Ga.Parts;
using Utils.Sortable;
using Utils.Sorter;

namespace SpongeTester
{
    [TestClass]
    public class GaFixture
    {
        [TestMethod]
        public void TestMakeGa()
        {
            const int seed = 1234;
            const int order = 16;
            const int stageCount = 20;
            const int sorterCount = 20;
            const int sortableCount = 20;

            var randy = Rando.Standard(seed);

            var randomSorterPool = randy.ToRandomSorterPool(
                order: order,
                stageCount: stageCount,
                sorterCount: sorterCount);


            var randomSortablePool = randy.ToRandomSortablePool(
                order: order,
                poolCount: sortableCount);


            var ga = new SortingGaPools(
                sorterPool:randomSorterPool, 
                sortablePool: randomSortablePool);

            var sorter0 = randomSorterPool[0];

            var sr = randomSortablePool.SelectMany(
                sb => randomSorterPool.Select(st=>st.Sort(sb)));

            var gar = new SortingResults(sr, false);

            var best = gar.SorterResults.Values.OrderBy(r => r.AverageSortedness);
        }

        [TestMethod]
        public void TestRefineGa()
        {
            const int seed = 5234;
            const int order = 48;
            const int stageCount = 20;
            const int sorterCount = 32;
            const int sortableCount = 128;
            const int selectionFactor = 4;
            const double replacementRate = 0.75;
            const int batchSize = 50;
            const int batchRounds = 40;

            var randy = Rando.Standard(seed);

            var randomSorterPool = randy.ToRandomSorterPool(
                order: order,
                stageCount: stageCount,
                sorterCount: sorterCount);

            var randomSortablePool = randy.ToRandomSortablePool(
                order: order,
                poolCount: sortableCount);

            var ga = new SortingGaPools(
                sorterPool: randomSorterPool,
                sortablePool: randomSortablePool);


            var curGa = ga;

            var SorterGa = new List<SortingGaPools>();
            var SortableGa = new List<SortingGaPools>();


            for (var j = 0; j < batchRounds; j++)
            {
                Console.WriteLine("____EvolveSorters______");

                for (var i = 0; i < batchSize; i++)
                {
                    var eval = curGa.Eval(false);
                    Console.WriteLine(eval.SorterResults.Average(sr => sr.Value.AverageSortedness));
                    curGa = curGa.EvolveSorters(eval.SorterResults, randy,  selectionFactor, StageReplacementMode.RandomRewire, true);
                }
                SorterGa.Add(curGa);

                Console.WriteLine("____EvolveSortables______");

                for (var i = 0; i < batchSize; i++)
                {
                    var eval = curGa.Eval(false);
                    Console.WriteLine(eval.SortableResults.Average(sr => sr.Value.AverageSortedness));
                    curGa = curGa.EvolveSortables(eval.SortableResults, randy, replacementRate, true);
                }
                SortableGa.Add(curGa);
            }



            Console.WriteLine("____Crosstab______");

            for (var j = 0; j < batchRounds; j++)
            {
                for (var i = 0; i < batchRounds; i++)
                {
                    ga = new SortingGaPools(
                        sorterPool: SorterGa[j].SorterPool,
                        sortablePool: SortableGa[i].SortablePool);


                    var eval = ga.Eval(false);


                    Console.WriteLine($"{j} {i} " +
                                      $"{eval.SorterResults.Average(sr => sr.Value.AverageSortedness)} " +
                                      $"{eval.SorterResults.Average(sr => sr.Value.StagesUsed) }"

                                      );
                }
            }
        }

        [TestMethod]
        public void TestRefineGa2()
        {
            const int seed = 234;
            const int order = 48;
            const int stageCount = 18;
            const int sorterCount = 32;
            const int sortableCount = 64;
            const int selectionFactor = 4;
            const int rounds = 12000;

            var randy = Rando.Standard(seed);

            var randomSorterPool = randy.ToRandomSorterPool(
                order: order,
                stageCount: stageCount,
                sorterCount: sorterCount);

            var randomSortablePool = randy.ToRandomSortablePool(
                order: order,
                poolCount: sortableCount);

            var ga = new SortingGaPools(
                sorterPool: randomSorterPool,
                sortablePool: randomSortablePool);


            var curGa = ga;


            for (var j = 0; j < rounds; j++)
            {
                var eval = curGa.Eval(false);
                Console.WriteLine(eval.SorterResults.Average(sr => sr.Value.AverageSortedness));
                curGa = curGa.EvolveSorters(eval.SorterResults, randy,  selectionFactor, StageReplacementMode.RandomRewire, true);
            }

        }


        [TestMethod]
        public void TestRefineGa3()
        {
            const int seed = 29734;
            const int order = 48;
            const int stageCount = 18;
            const int sorterCount = 32;
            const int sortableCount = 64;
            const int selectionFactor = 2;
            const double replacementRate = 0.5;
            const int days = 10;
            const int months = 3;
            const int years = 20;
            const int settle = 15;

            var randy = Rando.Standard(seed);

            var randomSorterPool = randy.ToRandomSorterPool(
                order: order,
                stageCount: stageCount,
                sorterCount: sorterCount);

            var randomSortablePool = randy.ToRandomSortablePool(
                order: order,
                poolCount: sortableCount);

            var ga = new SortingGaPools(
                sorterPool: randomSorterPool,
                sortablePool: randomSortablePool);


            var curGa = ga;
            var eval = curGa.Eval(false);
            var avgSorterPerformance = eval.SorterResults.Average(sr => sr.Value.AverageSortedness);
            var sorterOverlap = 0;
            var sortableOverlap = 0;
            var blSorters = curGa.SorterPool.Sorters;
            var blSortables = curGa.SortablePool.Sortables;

            for (var y = 0; y < years; y++)
            {
                blSorters = curGa.SorterPool.Sorters;
                blSortables = curGa.SortablePool.Sortables;

                for (var j = 0; j < months; j++)
                {
                    blSorters = curGa.SorterPool.Sorters;
                    blSortables = curGa.SortablePool.Sortables;
                    for (var i = 0; i < days; i++)
                    {
                        sortableOverlap = blSortables.KeyOverlap(curGa.SortablePool.Sortables);
                        sorterOverlap = blSorters.KeyOverlap(curGa.SorterPool.Sorters);
                        eval = curGa.Eval(false);
                        avgSorterPerformance = eval.SorterResults.Average(sr => sr.Value.AverageSortedness);
                        Console.WriteLine($"{avgSorterPerformance} {sorterOverlap} {sortableOverlap}");
                        curGa = curGa.EvolveSorters(eval.SorterResults, randy, selectionFactor, StageReplacementMode.RandomRewire, true);
                    }
                    blSorters = curGa.SorterPool.Sorters;
                    blSortables = curGa.SortablePool.Sortables;
                    for (var i = 0; i < days; i++)
                    {
                        sortableOverlap = blSortables.KeyOverlap(curGa.SortablePool.Sortables);
                        sorterOverlap = blSorters.KeyOverlap(curGa.SorterPool.Sorters);
                        eval = curGa.Eval(false);
                        avgSorterPerformance = eval.SorterResults.Average(sr => sr.Value.AverageSortedness);
                        Console.WriteLine($"{avgSorterPerformance} {sorterOverlap} {sortableOverlap}");
                        curGa = curGa.EvolveSortables(eval.SortableResults, randy, replacementRate, true);
                    }
                }


                for (var j = 0; j < months; j++)
                {
                    blSorters = curGa.SorterPool.Sorters;
                    blSortables = curGa.SortablePool.Sortables;
                    for (var i = 0; i < days * settle; i++)
                    {
                        sortableOverlap = blSortables.KeyOverlap(curGa.SortablePool.Sortables);
                        sorterOverlap = blSorters.KeyOverlap(curGa.SorterPool.Sorters);
                        eval = curGa.Eval(false);
                        avgSorterPerformance = eval.SorterResults.Average(sr => sr.Value.AverageSortedness);
                        Console.WriteLine($"{avgSorterPerformance} {sorterOverlap} {sortableOverlap}");
                        curGa = curGa.EvolveSorters(eval.SorterResults, randy, selectionFactor, StageReplacementMode.RandomRewire, true);
                    }
                }
            }


            for (var j = 0; j < 50; j++)
            {
                blSorters = curGa.SorterPool.Sorters;
                blSortables = curGa.SortablePool.Sortables;
                for (var i = 0; i < 60; i++)
                {
                    sortableOverlap = blSortables.KeyOverlap(curGa.SortablePool.Sortables);
                    sorterOverlap = blSorters.KeyOverlap(curGa.SorterPool.Sorters);
                    eval = curGa.Eval(false);
                    avgSorterPerformance = eval.SorterResults.Average(sr => sr.Value.AverageSortedness);
                    Console.WriteLine($"{avgSorterPerformance} {sorterOverlap} {sortableOverlap}");
                    curGa = curGa.EvolveSorters(eval.SorterResults, randy, selectionFactor, StageReplacementMode.RandomRewire, true);
                }
            }

        }


        [TestMethod]
        public void TestRefineBench()
        {
            const int seed = 35934;
            const int order = 28;
            const int stageCount = 16;
            const int sorterCount = 32;
            const int sortableCount = 32;
            const int selectionFactor = 4;
            const double replacementRate = 0.75;
            const int batchSize = 20;
            const int batchRounds = 20;

            var randy = Rando.Standard(seed);

            var randomSorterPool = randy.ToRandomSorterPool(
                order: order,
                stageCount: stageCount,
                sorterCount: sorterCount);

            var randomSortablePool = randy.ToRandomSortablePool(
                order: order,
                poolCount: sortableCount);

            var ga = new SortingGaPools(
                sorterPool: randomSorterPool,
                sortablePool: randomSortablePool);


            var curGa = ga;


            Stopwatch sw = new Stopwatch();
            sw.Start();


            for (var j = 0; j < batchRounds; j++)
            {
                for (var i = 0; i < batchSize; i++)
                {
                    var eval = curGa.Eval(false);
                    Console.WriteLine(eval.SorterResults.Average(sr => sr.Value.AverageSortedness));
                    curGa = curGa.EvolveSorters(eval.SorterResults, randy, selectionFactor, StageReplacementMode.RandomRewire, true);
                }

                for (var i = 0; i < batchSize; i++)
                {
                    var eval = curGa.Eval(false);
                    // Console.WriteLine(eval.SortableResults.Average(sr => sr.Value.AverageSortedness));
                    curGa = curGa.EvolveSortables(eval.SortableResults, randy, replacementRate, true);
                }
            }



            for (var j = 0; j < batchRounds; j++)
            {
                for (var i = 0; i < batchSize; i++)
                {
                    var eval = curGa.Eval(false);
                    Console.WriteLine(eval.SorterResults.Average(sr => sr.Value.AverageSortedness));
                    curGa = curGa.EvolveSorters(eval.SorterResults, randy, selectionFactor, StageReplacementMode.RandomRewire, true);
                }
            }

            sw.Stop();
            Console.WriteLine("\nElapsed={0}", sw.Elapsed);

        }


    }
}
