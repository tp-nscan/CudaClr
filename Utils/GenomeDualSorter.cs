using System;
using System.Collections.Generic;
using System.Linq;

namespace Utils
{
    public interface ISorterGenome
    {
        Guid Id { get; }
        int StageCount { get; }
        ISorterStage ChromoA(int index);
        ISorterStage ChromoB(int index);
        IEnumerable<ISorterStage> ChromosomeA { get; }
        IEnumerable<ISorterStage> ChromosomeB { get; }
    }

    public class GenomeDualSorter : ISorterGenome
    {
        public GenomeDualSorter(Guid id, 
                                IEnumerable<ISorterStage> chromA,
                                IEnumerable<ISorterStage> chromB, 
                                IEnumerable<bool> choices)
        {
            Id = id;

            var curStageNumber = 0;
            _chromosomeA = chromA.Select(
                s => new SorterStage(
                    order: s.Order,
                    terms: s.GetMap(),
                    stageNumber: curStageNumber++
                ) as ISorterStage).ToList();

            curStageNumber = 0;
            _chromosomeB = chromB.Select(
                s => new SorterStage(
                    order: s.Order,
                    terms: s.GetMap(),
                    stageNumber: curStageNumber++
                ) as ISorterStage).ToList();

            _choices = choices.ToList();
        }

        readonly List<bool> _choices;
        readonly List<ISorterStage> _chromosomeA;
        readonly List<ISorterStage> _chromosomeB;

        public Guid Id { get; }

        public int StageCount => _chromosomeA.Count();

        public ISorterStage ChromoA(int index) => _chromosomeA[index];
        public ISorterStage ChromoB(int index) => _chromosomeB[index];
        public bool Choice(int index) => _choices[index];

        public IEnumerable<ISorterStage> ChromosomeA => _chromosomeA;
        public IEnumerable<ISorterStage> ChromosomeB => _chromosomeB;
        public IEnumerable<bool> Choices => _choices;
    }

    public static class DualSorterGenomeExt
    {
        public static GenomeDualSorter ToDualSorterGenome(this IRando randy, int order, int stageCount)
        {
            return new GenomeDualSorter(
                id: Guid.NewGuid(),
                chromA: Enumerable.Range(0, stageCount)
                    .Select(i => randy.RandomFullSorterStage(order, i)),
                chromB: Enumerable.Range(0, stageCount)
                    .Select(i => randy.RandomFullSorterStage(order, i)),
                choices: Enumerable.Range(0, stageCount)
                    .Select(i => randy.NextBool(0.5))
                );
        }

        public static GenomeDualSorter ToDualSorterGenomeUniform(this IRando randy, int order, int stageCount)
        {
            var ch = Enumerable.Range(0, stageCount)
                .Select(i => randy.RandomFullSorterStage(order, i)).ToList();
            return new GenomeDualSorter(
                id: Guid.NewGuid(),
                chromA: ch,
                chromB: ch,
                choices: Enumerable.Range(0, stageCount)
                    .Select(i => randy.NextBool(0.5))
                );
        }

        public static GenomeDualSorter RecombineI(this GenomeDualSorter genomeDualSorter, IRando randy)
        {
            var recombines = randy.Recombo(genomeDualSorter.ChromosomeA.ToList(), 
                                          genomeDualSorter.ChromosomeB.ToList());

            return new GenomeDualSorter(
                id: Guid.NewGuid(), 
                chromA: recombines.Item1.ToList(), 
                chromB: recombines.Item2.ToList(),
                choices: genomeDualSorter.Choices);
        }

        public static Tuple<GenomeDualSorter, GenomeDualSorter> Recombine(
            this GenomeDualSorter genomeDualSorterA,
            GenomeDualSorter genomeDualSorterB,
            IRando randy)
        {
            var recombin1 = randy.Recombo(
                genomeDualSorterA.ChromosomeA.ToList(),
                genomeDualSorterB.ChromosomeA.ToList());

            var recombin2 = randy.Recombo(
                genomeDualSorterA.ChromosomeB.ToList(),
                genomeDualSorterB.ChromosomeB.ToList());

            if (randy.NextBool(0.5))
            {
                return new Tuple<GenomeDualSorter, GenomeDualSorter>(
                    item1: new GenomeDualSorter(
                        id: Guid.NewGuid(),
                        chromA: recombin1.Item1,
                        chromB: recombin1.Item2,
                        choices: genomeDualSorterA.Choices),
                    item2: new GenomeDualSorter(
                        id: Guid.NewGuid(),
                        chromA: recombin2.Item1,
                        chromB: recombin2.Item2,
                        choices: genomeDualSorterA.Choices)
                    );
            }

            return new Tuple<GenomeDualSorter, GenomeDualSorter>(
                item1: new GenomeDualSorter(
                    id: Guid.NewGuid(),
                    chromA: recombin1.Item2,
                    chromB: recombin1.Item1,
                    choices: genomeDualSorterA.Choices),
                item2: new GenomeDualSorter(
                    id: Guid.NewGuid(),
                    chromA: recombin2.Item2,
                    chromB: recombin2.Item1,
                    choices: genomeDualSorterA.Choices)
            );
        }


        public static GenomeDualSorter MakeDualSorterGenome(
            Guid id,
            IEnumerable<ISorterStage> chromA,
            IEnumerable<ISorterStage> chromB,
            IEnumerable<bool> choices)
        {
            return new GenomeDualSorter(
                id: id,
                chromA: chromA,
                chromB: chromB,
                choices:choices);
        }

        public static GenomeDualSorter Mutate(this 
            GenomeDualSorter genomeDualSorter, IRando randy)
        {
            var mutantIndex = randy.NextInt(genomeDualSorter.StageCount);
            var mutantChoiceIndex = randy.NextInt(genomeDualSorter.StageCount);
            var newChromA = genomeDualSorter.ChromosomeA;
            var newChromB = genomeDualSorter.ChromosomeB;

            var newChoices = genomeDualSorter.Choices;

            //if (randy.NextBool(0.1))
            //{
            //    newChoices = genomeDualSorter.Choices.ReplaceAtIndex(mutantChoiceIndex,
            //        !genomeDualSorter.Choice(mutantChoiceIndex));
            //}

            if (randy.NextBool(0.5))
            {
                var mutantStage = randy.MutateSorterStage(genomeDualSorter.ChromoA(mutantIndex));
                newChromA = newChromA.ReplaceAtIndex(mutantIndex, mutantStage);
            }
            else
            {
                var mutantStage = randy.MutateSorterStage(genomeDualSorter.ChromoB(mutantIndex));
                newChromB = newChromB.ReplaceAtIndex(mutantIndex, mutantStage);
            }

            return new GenomeDualSorter(
                id: Guid.NewGuid(),
                chromA: newChromA,
                chromB: newChromB,
                choices: newChoices);
        }

        public static IEnumerable<ISorter> MakePhenotypes(this GenomeDualSorter genomeDualSorter,
            IRando randy, int count)
        {
            for (var i = 0; i < count; i++)
            {
                yield return Enumerable.Range(0, genomeDualSorter.StageCount)
                    .Select(d => genomeDualSorter.Choice(d)
                        ? genomeDualSorter.ChromoA(d)
                        : genomeDualSorter.ChromoB(d))
                    .MakeSorter(id: Guid.NewGuid(), genomeId: genomeDualSorter.Id);
            }
        }

        //public static IEnumerable<ISorter> MakePhenotypes(this GenomeDualSorter genomeDualSorter,
        //    IRando randy, int count)
        //{
        //    for (var i = 0; i < count; i++)
        //    {
        //        yield return randy.ToIndexedBoolEnumerator(0.5)
        //            .Take(genomeDualSorter.StageCount)
        //            .Select(b => b.Item2
        //                ? genomeDualSorter.ChromoA(b.Item1)
        //                : genomeDualSorter.ChromoB(b.Item1))
        //            .MakeSorter(id: Guid.NewGuid(), genomeId: genomeDualSorter.Id);
        //    }
        //}


    }
}