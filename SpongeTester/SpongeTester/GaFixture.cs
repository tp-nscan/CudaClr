using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Utils;

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

            var randomSorterPool = randy.RandomSorterPool(
                order: order,
                stageCount: stageCount,
                sorterCount: sorterCount);


            var randomSortablePool = randy.RandomSortablePool(
                order: order,
                sortableCount: sortableCount);


            var ga = new Ga(
                sorterPool:randomSorterPool, 
                sortablePool: randomSortablePool);

            var sorter0 = randomSorterPool[0];

            var sr = randomSortablePool.SelectMany(
                sb => randomSorterPool.Select(st=>st.Sort(sb)));

            var gar = new GaResult(ga, sr, false);

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

            var randomSorterPool = randy.RandomSorterPool(
                order: order,
                stageCount: stageCount,
                sorterCount: sorterCount);

            var randomSortablePool = randy.RandomSortablePool(
                order: order,
                sortableCount: sortableCount);

            var ga = new Ga(
                sorterPool: randomSorterPool,
                sortablePool: randomSortablePool);


            var curGa = ga;

            List<Ga> SorterGa = new List<Ga>();
            List<Ga> SortableGa = new List<Ga>();


            for (var j = 0; j < batchRounds; j++)
            {
                Console.WriteLine("____EvolveSorters______");

                for (var i = 0; i < batchSize; i++)
                {
                    var eval = curGa.Eval(false);
                    Console.WriteLine(eval.SorterResults.Average(sr => sr.Value.AverageSortedness));
                    curGa = eval.EvolveSorters(randy, selectionFactor);
                }
                SorterGa.Add(curGa);

                Console.WriteLine("____EvolveSortables______");

                for (var i = 0; i < batchSize; i++)
                {
                    var eval = curGa.Eval(false);
                    Console.WriteLine(eval.SortableResults.Average(sr => sr.Value.AverageSortedness));
                    curGa = eval.EvolveSortables(randy, replacementRate);
                }
                SortableGa.Add(curGa);

            }



            Console.WriteLine("____Crosstab______");

            for (var j = 0; j < batchRounds; j++)
            {
                for (var i = 0; i < batchRounds; i++)
                {
                    ga = new Ga(
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
            const int seed = 5234;
            const int order = 48;
            const int stageCount = 16;
            const int sorterCount = 32;
            const int sortableCount = 32;
            const int selectionFactor = 4;
            const double replacementRate = 0.75;
            const int batchSize = 40;
            const int batchRounds = 300;

            var randy = Rando.Standard(seed);

            var randomSorterPool = randy.RandomSorterPool(
                order: order,
                stageCount: stageCount,
                sorterCount: sorterCount);

            var randomSortablePool = randy.RandomSortablePool(
                order: order,
                sortableCount: sortableCount);

            var ga = new Ga(
                sorterPool: randomSorterPool,
                sortablePool: randomSortablePool);


            var curGa = ga;


            for (var j = 0; j < batchRounds; j++)
            {
                for (var i = 0; i < batchSize; i++)
                {
                    var eval = curGa.Eval(false);
                    Console.WriteLine(eval.SorterResults.Average(sr => sr.Value.AverageSortedness));
                    curGa = eval.EvolveSorters(randy, selectionFactor);
                }
            }

        }


        [TestMethod]
        public void TestRefineGa3()
        {
            const int seed = 3934;
            const int order = 48;
            const int stageCount = 16;
            const int sorterCount = 64;
            const int sortableCount = 64;
            const int selectionFactor = 4;
            const double replacementRate = 0.5;
            const int days = 20;
            const int months = 3;
            const int years = 30;
            const int settle = 10;

            var randy = Rando.Standard(seed);

            var randomSorterPool = randy.RandomSorterPool(
                order: order,
                stageCount: stageCount,
                sorterCount: sorterCount);

            var randomSortablePool = randy.RandomSortablePool(
                order: order,
                sortableCount: sortableCount);

            var ga = new Ga(
                sorterPool: randomSorterPool,
                sortablePool: randomSortablePool);


            var curGa = ga;

            for (var y = 0; y < years; y++)
            {
                for (var j = 0; j < months; j++)
                {
                    for (var i = 0; i < days; i++)
                    {
                        var eval = curGa.Eval(false);
                        Console.WriteLine(eval.SorterResults.Average(sr => sr.Value.AverageSortedness));
                        curGa = eval.EvolveSorters(randy, selectionFactor);
                    }

                    for (var i = 0; i < days; i++)
                    {
                        var eval = curGa.Eval(false);
                        // Console.WriteLine(eval.SortableResults.Average(sr => sr.Value.AverageSortedness));
                        curGa = eval.EvolveSortables(randy, replacementRate);
                    }
                }


                for (var j = 0; j < months * settle; j++)
                {
                    for (var i = 0; i < days; i++)
                    {
                        var eval = curGa.Eval(false);
                        Console.WriteLine(eval.SorterResults.Average(sr => sr.Value.AverageSortedness));
                        curGa = eval.EvolveSorters(randy, selectionFactor);
                    }
                }
            }

        }


        [TestMethod]
        public void TestRefineBench()
        {
            const int seed = 35934;
            const int order = 48;
            const int stageCount = 16;
            const int sorterCount = 32;
            const int sortableCount = 32;
            const int selectionFactor = 4;
            const double replacementRate = 0.75;
            const int batchSize = 100;
            const int batchRounds = 10;

            var randy = Rando.Standard(seed);

            var randomSorterPool = randy.RandomSorterPool(
                order: order,
                stageCount: stageCount,
                sorterCount: sorterCount);

            var randomSortablePool = randy.RandomSortablePool(
                order: order,
                sortableCount: sortableCount);

            var ga = new Ga(
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
                    curGa = eval.EvolveSorters(randy, selectionFactor);
                }

                for (var i = 0; i < batchSize; i++)
                {
                    var eval = curGa.Eval(false);
                    // Console.WriteLine(eval.SortableResults.Average(sr => sr.Value.AverageSortedness));
                    curGa = eval.EvolveSortables(randy, replacementRate);
                }
            }



            for (var j = 0; j < batchRounds; j++)
            {
                for (var i = 0; i < batchSize; i++)
                {
                    var eval = curGa.Eval(false);
                    Console.WriteLine(eval.SorterResults.Average(sr => sr.Value.AverageSortedness));
                    curGa = eval.EvolveSorters(randy, selectionFactor);
                }
            }


            sw.Stop();

            Console.WriteLine("Elapsed={0}", sw.Elapsed);


        }


    }
}
