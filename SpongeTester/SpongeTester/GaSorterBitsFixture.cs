using System;
using System.Diagnostics;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Utils;
using Utils.Ga;
using Utils.Genome;
using Utils.Sortable;
using Utils.Sorter;

namespace SpongeTester
{
    [TestClass]
    public class GaSorterBitsFixture
    {

        public double Mutato1(double mutationRate, double step)
        {
            return mutationRate * (20.0/step);
        }

        public double Mutato2(double mutationRate, double step)
        {
            var mr = mutationRate * (1.0 / Math.Log(step + 10));
            return mr;
        }

        public double Mutato(double step)
        {
            double uni = 4.69219E-05;
            double a = 0.1;
            double b = 0.5;
            double c = 1.0;
            return uni * (a + b / (Math.Log(step + 10) + c));
        }

        [TestMethod]
        public void TestRefineBench2()
        {
            const int seed = 3196;
            const int order = 128;
            const int stageCount = 12;
            const int sorterCount = 64;
            const int sortableCount = 64;
            const int selectionFactor = 4;
            const double replacementRate = 0.75;
            const int batchSize = 25;
            const int batchRounds = 2000;

            double step = 1.0;

            var randy = Rando.Standard(seed);

            var randomSorterPool = randy.ToSorterPool(
                order: order,
                stageCount: stageCount,
                sorterCount: sorterCount);

            var randomSortablePool = randy.RandomSortablePool(
                order: order,
                poolCount: sortableCount);

            var ga = new Ga(
                sorterPool: randomSorterPool,
                sortablePool: randomSortablePool);


            var genomePoolSorterBits = randy.ToGenomePoolSorterBits(
                order: order,
                stageCount: stageCount,
                poolCount: sorterCount);


            var gaB = new GaSorterBits(
                genomePoolSorterBits: genomePoolSorterBits,
                sortablePool: randomSortablePool,
                randy: randy);

            var curGa = ga;
            var curGaB = gaB;


            Stopwatch sw = new Stopwatch();
            sw.Start();


            for (var j = 0; j < batchRounds; j++)
            {
                for (var i = 0; i < batchSize; i++)
                {
                   // var evalB = curGaB.Eval(false);
                    var eval = curGa.Eval(false);

                    var avgStr = eval.SorterResults.Average(sr => sr.Value.AverageSortedness);
                    /// var avgStrB = evalB.SorterResults.Average(sr => sr.Value.AverageSortedness);
                    Console.WriteLine($"{avgStr}");

                    curGa = curGa.EvolveSorters(eval.SorterResults, randy, selectionFactor, StageReplacementMode.RCTC, true);
                   // curGaB = evalB.EvolveSorters(randy, selectionFactor, Mutato( step++));
                }

                for (var i = 0; i < batchSize; i++)
                {
                    var eval = curGa.Eval(false);
                   // var evalB = curGaB.Eval(false);


                    curGa = curGa.EvolveSortables(eval.SortableResults, randy, replacementRate, true);
                   // curGaB = evalB.EvolveSortables(randy, replacementRate);
                }
            }



            for (var j = 0; j < batchRounds; j++)
            {
                for (var i = 0; i < batchSize; i++)
                {
                   // var evalB = curGaB.Eval(false);
                    var eval = curGa.Eval(false);

                    var avgStr = eval.SorterResults.Average(sr => sr.Value.AverageSortedness);
                  //  var avgStrB = evalB.SorterResults.Average(sr => sr.Value.AverageSortedness);
                    Console.WriteLine($"{avgStr}");

                    curGa = curGa.EvolveSorters(eval.SorterResults, randy, selectionFactor, StageReplacementMode.RCTC, true);
                    //curGaB = evalB.EvolveSorters(randy, selectionFactor, Mutato(step++));
                }
            }

            sw.Stop();
            Console.WriteLine("\nElapsed={0}", sw.Elapsed);

        }

        [TestMethod]
        public void TestRefineBench3()
        {
            const int seed = 3963;
            const int order = 64;
            const int stageCount = 12;
            const int sorterCount = 128;
            const int sortableCount = 256;
            const int selectionFactor = 4;
            const double replacementRate = 0.75;
            const int sorterRounds = 10;
            const int sortableRounds = 10;
            const int batchRounds = 60;

            double step = 1.0;

            var randy = Rando.Standard(seed);

            var randomSorterPool = randy.ToSorterPool(
                order: order,
                stageCount: stageCount,
                sorterCount: sorterCount);

            var randomSortablePool = randy.RandomSortablePool(
                order: order,
                poolCount: sortableCount);

            var ga = new Ga(
                sorterPool: randomSorterPool,
                sortablePool: randomSortablePool);


            var gaB = new Ga(
                sorterPool: randomSorterPool,
                sortablePool: randomSortablePool);


            var gaC = new Ga(
                sorterPool: randomSorterPool,
                sortablePool: randomSortablePool);

            var gaD = new Ga(
                sorterPool: randomSorterPool,
                sortablePool: randomSortablePool);

            var curGa = ga;
            var curGaB = gaB;
            var curGaC = gaC;
            var curGaD = gaD;


            Stopwatch sw = new Stopwatch();
            sw.Start();


            for (var j = 0; j < batchRounds; j++)
            {
                for (var i = 0; i < sorterRounds; i++)
                {
                    var evalA = curGa.Eval(false);
                    var evalB = curGaB.Eval(false);
                    var evalC = curGaC.Eval(false);
                    var evalD = curGaD.Eval(false);

                    var avgStr = evalA.SorterResults.Average(sr => sr.Value.AverageSortedness);
                    var avgStrB = evalB.SorterResults.Average(sr => sr.Value.AverageSortedness);
                    var avgStrC = evalC.SorterResults.Average(sr => sr.Value.AverageSortedness);
                    var avgStrD = evalD.SorterResults.Average(sr => sr.Value.AverageSortedness);

                    Console.WriteLine($"{avgStr} {avgStrB} {avgStrC} {avgStrD}");

                    curGa = curGa.EvolveSorters(evalA.SorterResults, randy, selectionFactor, StageReplacementMode.RandomReplace, true);
                    curGaB = curGaB.EvolveSorters(evalB.SorterResults, randy, selectionFactor, StageReplacementMode.RandomReplace, true);
                    curGaC = curGaC.EvolveSorters(evalC.SorterResults, randy, selectionFactor, StageReplacementMode.RandomReplace, true);
                    curGaD = curGaD.EvolveSorters(evalD.SorterResults, randy, selectionFactor, StageReplacementMode.RandomReplace, true);

                }
                
                for (var i = 0; i < sortableRounds; i++)
                {
                    var evalA = curGa.Eval(false);
                    var evalB = curGaB.Eval(false);
                    var evalC = curGaC.Eval(false);
                    var evalD = curGaD.Eval(false);


                    var avgStr = evalA.SorterResults.Average(sr => sr.Value.AverageSortedness);
                    var avgStrB = evalB.SorterResults.Average(sr => sr.Value.AverageSortedness);
                    var avgStrC = evalC.SorterResults.Average(sr => sr.Value.AverageSortedness);
                    var avgStrD = evalD.SorterResults.Average(sr => sr.Value.AverageSortedness);
                    Console.WriteLine($"{avgStr} {avgStrB} {avgStrC} {avgStrD}");

                    curGa = curGa.EvolveSortables(evalA.SortableResults, randy, replacementRate, true);
                    curGaB = curGaB.EvolveSortables(evalB.SortableResults, randy, replacementRate, true);
                    curGaC = curGaC.EvolveSortables(evalC.SortableResults, randy, replacementRate, true);
                }
            }



            for (var j = 0; j < 10; j++)
            {
                for (var i = 0; i < sorterRounds; i++)
                {
                    var eval = curGa.Eval(false);
                    var evalB = curGaB.Eval(false);
                    var evalC = curGaC.Eval(false);
                    var evalD = curGaD.Eval(false);

                    var avgStr = eval.SorterResults.Average(sr => sr.Value.AverageSortedness);
                    var avgStrB = evalB.SorterResults.Average(sr => sr.Value.AverageSortedness);
                    var avgStrC = evalC.SorterResults.Average(sr => sr.Value.AverageSortedness);
                    var avgStrD = evalD.SorterResults.Average(sr => sr.Value.AverageSortedness);
                    Console.WriteLine($"{avgStr} {avgStrB} {avgStrC} {avgStrD}");

                    curGa = curGa.EvolveSorters(eval.SorterResults, randy, selectionFactor, StageReplacementMode.RandomReplace, true);
                    curGaB = curGaB.EvolveSorters(evalB.SorterResults, randy, selectionFactor, StageReplacementMode.RandomReplace, true);
                    curGaC = curGaC.EvolveSorters(evalC.SorterResults, randy, selectionFactor, StageReplacementMode.RandomReplace, true);
                    curGaD = curGaD.EvolveSorters(evalD.SorterResults, randy, selectionFactor, StageReplacementMode.RandomReplace, true);
                }
            }

            sw.Stop();
            Console.WriteLine("\nElapsed={0}", sw.Elapsed);

        }

        [TestMethod]
        public void TestRefineBench4()
        {
            const int seed = 3963;
            const int order = 80;
            const int stageCount = 12;
            const int sorterCount = 4;
            const int sortableCount = 256;
            const int selectionFactor = 4;
            const double replacementRate = 0.75;
            const int sorterRounds = 10;
            const int sortableRounds = 10;
            const int batchRounds = 1200;

            double step = 1.0;

            var randy = Rando.Standard(seed);

            var randomSorterPoolA = randy.ToSorterPool(
                order: order,
                stageCount: stageCount,
                sorterCount: sorterCount);

            var randomSorterPoolB = randy.ToSorterPool(
                order: order,
                stageCount: stageCount,
                sorterCount: sorterCount * 2);

            var randomSorterPoolC = randy.ToSorterPool(
                order: order,
                stageCount: stageCount,
                sorterCount: sorterCount * 4);

            var randomSorterPoolD = randy.ToSorterPool(
                order: order,
                stageCount: stageCount,
                sorterCount: sorterCount * 8);



            var randomSortablePool = randy.RandomSortablePool(
                order: order,
                poolCount: sortableCount);

            var ga = new Ga(
                sorterPool: randomSorterPoolA,
                sortablePool: randomSortablePool);


            var gaB = new Ga(
                sorterPool: randomSorterPoolB,
                sortablePool: randomSortablePool);


            var gaC = new Ga(
                sorterPool: randomSorterPoolC,
                sortablePool: randomSortablePool);

            var gaD = new Ga(
                sorterPool: randomSorterPoolD,
                sortablePool: randomSortablePool);

            var curGa = ga;
            var curGaB = gaB;
            var curGaC = gaC;
            var curGaD = gaD;


            Stopwatch sw = new Stopwatch();
            sw.Start();


            for (var j = 0; j < batchRounds; j++)
            {
                for (var i = 0; i < sorterRounds; i++)
                {
                    var eval = curGa.Eval(false);
                    var evalB = curGaB.Eval(false);
                    var evalC = curGaC.Eval(false);
                    var evalD = curGaD.Eval(false);

                    var avgStr = eval.SorterResults.Average(sr => sr.Value.AverageSortedness);
                    var avgStrB = evalB.SorterResults.Average(sr => sr.Value.AverageSortedness);
                    var avgStrC = evalC.SorterResults.Average(sr => sr.Value.AverageSortedness);
                    var avgStrD = evalD.SorterResults.Average(sr => sr.Value.AverageSortedness);

                    Console.WriteLine($"{avgStr} {avgStrB} {avgStrC} {avgStrD}");

                    curGa = curGa.EvolveSorters(eval.SorterResults, randy, selectionFactor, StageReplacementMode.RandomReplace, true);
                    curGaB = curGaB.EvolveSorters(evalB.SorterResults, randy, selectionFactor, StageReplacementMode.RandomReplace, true);
                    curGaC = curGaC.EvolveSorters(evalC.SorterResults, randy, selectionFactor, StageReplacementMode.RandomReplace, true);
                    curGaD = curGaD.EvolveSorters(evalD.SorterResults, randy, selectionFactor, StageReplacementMode.RandomReplace, true);

                }

                //for (var i = 0; i < sortableRounds; i++)
                //{
                //    var eval = curGa.Eval(false);
                //    var evalB = curGaB.Eval(false);
                //    var evalC = curGaC.Eval(false);
                //    var evalD = curGaD.Eval(false);


                //    var avgStr = eval.SorterResults.Average(sr => sr.Value.AverageSortedness);
                //    var avgStrB = evalB.SorterResults.Average(sr => sr.Value.AverageSortedness);
                //    var avgStrC = evalC.SorterResults.Average(sr => sr.Value.AverageSortedness);
                //    var avgStrD = evalD.SorterResults.Average(sr => sr.Value.AverageSortedness);
                //    Console.WriteLine($"{avgStr} {avgStrB} {avgStrC} {avgStrD}");

                //    curGa = eval.EvolveSortables(randy, replacementRate);
                //    curGaB = evalB.EvolveSortables(randy, replacementRate);
                //    curGaC = evalC.EvolveSortables(randy, replacementRate);
                //    curGaD = evalD.EvolveSortables(randy, replacementRate);
                //}

            }



            for (var j = 0; j < 10; j++)
            {
                for (var i = 0; i < sorterRounds; i++)
                {
                    var eval = curGa.Eval(false);
                    var evalB = curGaB.Eval(false);
                    var evalC = curGaC.Eval(false);
                    var evalD = curGaD.Eval(false);

                    var avgStr = eval.SorterResults.Average(sr => sr.Value.AverageSortedness);
                    var avgStrB = evalB.SorterResults.Average(sr => sr.Value.AverageSortedness);
                    var avgStrC = evalC.SorterResults.Average(sr => sr.Value.AverageSortedness);
                    var avgStrD = evalD.SorterResults.Average(sr => sr.Value.AverageSortedness);
                    Console.WriteLine($"{avgStr} {avgStrB} {avgStrC} {avgStrD}");

                    curGa = curGa.EvolveSorters(eval.SorterResults, randy, selectionFactor, StageReplacementMode.RandomReplace, true);
                    curGaB = curGaB.EvolveSorters(evalB.SorterResults, randy, selectionFactor, StageReplacementMode.RandomReplace, true);
                    curGaC = curGaC.EvolveSorters(evalC.SorterResults, randy, selectionFactor, StageReplacementMode.RandomReplace, true);
                    curGaD = curGaD.EvolveSorters(evalD.SorterResults, randy, selectionFactor, StageReplacementMode.RandomReplace, true);
                }
            }

            sw.Stop();
            Console.WriteLine("\nElapsed={0}", sw.Elapsed);

        }




        [TestMethod]
        public void TestRefineBench5()
        {
            const int seed = 3963;
            const int order = 64;
            const int stageCount = 12;
            const int sorterCount = 64;
            const int sortableCount = 64;
            const int selectionFactor = 4;
            const double replacementRate = 0.75;
            const int sorterRounds = 25;
            const int sortableRounds = 25;
            const int batchRounds = 50;

            //const int seed = 3963;
            //const int order = 80;
            //const int stageCount = 12;
            //const int sorterCount = 64;
            //const int sortableCount = 128;
            //const int selectionFactor = 4;
            //const double replacementRate = 0.75;
            //const int sorterRounds = 30;
            //const int sortableRounds = 10;
            //const int batchRounds = 1000;

            double step = 1.0;

            var randy = Rando.Standard(seed);

            var randomSorterPoolA = randy.ToSorterPool(
                order: order,
                stageCount: stageCount,
                sorterCount: sorterCount);

            var randomSorterPoolB = randy.ToSorterPool(
                order: order,
                stageCount: stageCount,
                sorterCount: sorterCount);

            var randomSorterPoolC = randy.ToSorterPool(
                order: order,
                stageCount: stageCount,
                sorterCount: sorterCount);

            var randomSorterPoolD = randy.ToSorterPool(
                order: order,
                stageCount: stageCount,
                sorterCount: sorterCount);



            var randomSortablePool = randy.RandomSortablePool(
                order: order,
                poolCount: sortableCount);

            var ga = new Ga(
                sorterPool: randomSorterPoolA,
                sortablePool: randomSortablePool);


            var gaB = new Ga(
                sorterPool: randomSorterPoolB,
                sortablePool: randomSortablePool);


            var gaC = new Ga(
                sorterPool: randomSorterPoolC,
                sortablePool: randomSortablePool);

            var gaD = new Ga(
                sorterPool: randomSorterPoolD,
                sortablePool: randomSortablePool);

            var curGa = ga;
            var curGaB = gaB;
            var curGaC = gaC;
            var curGaD = gaD;


            Stopwatch sw = new Stopwatch();
            sw.Start();


            for (var j = 0; j < batchRounds; j++)
            {
                for (var i = 0; i < sorterRounds; i++)
                {
                    var evalA = curGa.Eval(false);
                    var evalB = curGaB.Eval(false);
                    var evalC = curGaC.Eval(false);
                    var evalD = curGaD.Eval(false);

                    var avgStr = evalA.SorterResults.Average(sr => sr.Value.AverageSortedness);
                    var avgStrB = evalB.SorterResults.Average(sr => sr.Value.AverageSortedness);
                    var avgStrC = evalC.SorterResults.Average(sr => sr.Value.AverageSortedness);
                    var avgStrD = evalD.SorterResults.Average(sr => sr.Value.AverageSortedness);

                    Console.WriteLine($"{avgStr} {avgStrB} {avgStrC} {avgStrD}");

                    curGa = curGa.EvolveSorters(evalA.SorterResults, randy, selectionFactor, StageReplacementMode.RandomReplace, true);
                    curGaB = curGaB.EvolveSorters(evalB.SorterResults, randy, selectionFactor, StageReplacementMode.RandomReplace, true);
                    curGaC = curGaC.EvolveSorters(evalC.SorterResults, randy, selectionFactor, StageReplacementMode.RandomReplace, true);
                    curGaD = curGaD.EvolveSorters(evalD.SorterResults, randy, selectionFactor, StageReplacementMode.RandomReplace, true);

                }

                for (var i = 0; i < sortableRounds; i++)
                {
                    var evalA = curGa.Eval(false);
                    var evalB = curGaB.Eval(false);
                    var evalC = curGaC.Eval(false);
                    var evalD = curGaD.Eval(false);


                    var avgStr = evalA.SorterResults.Average(sr => sr.Value.AverageSortedness);
                    var avgStrB = evalB.SorterResults.Average(sr => sr.Value.AverageSortedness);
                    var avgStrC = evalC.SorterResults.Average(sr => sr.Value.AverageSortedness);
                    var avgStrD = evalD.SorterResults.Average(sr => sr.Value.AverageSortedness);
                    Console.WriteLine($"{avgStr} {avgStrB} {avgStrC} {avgStrD}");

                    curGa = curGa.EvolveSortables(evalA.SortableResults, randy, replacementRate, true);
                    curGaB = curGaB.EvolveSortables(evalB.SortableResults, randy, replacementRate, true);
                    curGaC = curGaC.EvolveSortables(evalC.SortableResults, randy, replacementRate, true);
                }

            }



            for (var j = 0; j < 10; j++)
            {
                for (var i = 0; i < sorterRounds; i++)
                {
                    var evalA = curGa.Eval(false);
                    var evalB = curGaB.Eval(false);
                    var evalC = curGaC.Eval(false);
                    var evalD = curGaD.Eval(false);

                    var avgStr = evalA.SorterResults.Average(sr => sr.Value.AverageSortedness);
                    var avgStrB = evalB.SorterResults.Average(sr => sr.Value.AverageSortedness);
                    var avgStrC = evalC.SorterResults.Average(sr => sr.Value.AverageSortedness);
                    var avgStrD = evalD.SorterResults.Average(sr => sr.Value.AverageSortedness);
                    Console.WriteLine($"{avgStr} {avgStrB} {avgStrC} {avgStrD}");

                    curGa = curGa.EvolveSorters(evalA.SorterResults, randy, selectionFactor, StageReplacementMode.RandomReplace, true);
                    curGaB = curGaB.EvolveSorters(evalB.SorterResults, randy, selectionFactor, StageReplacementMode.RandomReplace, true);
                    curGaC = curGaC.EvolveSorters(evalC.SorterResults, randy, selectionFactor, StageReplacementMode.RandomReplace, true);
                    curGaD = curGaD.EvolveSorters(evalD.SorterResults, randy, selectionFactor, StageReplacementMode.RandomReplace, true);
                }
            }

            sw.Stop();
            Console.WriteLine("\nElapsed={0}", sw.Elapsed);

        }





    }
}
