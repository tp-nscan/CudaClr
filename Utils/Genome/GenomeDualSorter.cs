using System;
using System.Collections.Generic;
using System.Linq;
using Utils.Sorter;

namespace Utils.Genome
{
    public interface IGenomeDualSorter : IGuid
    {
        uint StageCount { get; }
        ISorterStage ChromoA(uint index);
        ISorterStage ChromoB(uint index);
        IEnumerable<ISorterStage> ChromosomeA { get; }
        IEnumerable<ISorterStage> ChromosomeB { get; }
    }

    public class GenomeDualSorter : IGenomeDualSorter
    {
        public GenomeDualSorter(Guid id, 
                                IEnumerable<ISorterStage> chromA,
                                IEnumerable<ISorterStage> chromB, 
                                IEnumerable<bool> choices)
        {
            Id = id;

            var curStageNumber = 0u;
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

        public uint StageCount => (uint) _chromosomeA.Count();

        public ISorterStage ChromoA(uint index) => _chromosomeA[(int) index];
        public ISorterStage ChromoB(uint index) => _chromosomeB[(int) index];
        public bool Choice(uint index) => _choices[(int) index];

        public IEnumerable<ISorterStage> ChromosomeA => _chromosomeA;
        public IEnumerable<ISorterStage> ChromosomeB => _chromosomeB;
        public IEnumerable<bool> Choices => _choices;
    }

    public static class DualSorterGenomeExt
    {
        public static GenomeDualSorter ToDualSorterGenome(this IRando randy, uint order, uint stageCount)
        {
            return new GenomeDualSorter(
                id: Guid.NewGuid(),
                chromA: 0u.CountUp(stageCount)
                    .Select(i => randy.ToFullSorterStage(order, i)),
                chromB: 0u.CountUp(stageCount)
                    .Select(i => randy.ToFullSorterStage(order, i)),
                choices: 0u.CountUp(stageCount)
                    .Select(i => randy.NextBool(0.5))
                );
        }

        public static GenomeDualSorter ToDualSorterGenomeUniform(this IRando randy, uint order, uint stageCount)
        {
            var ch = 0u.CountUp(stageCount)
                .Select(i => randy.ToFullSorterStage(order, i)).ToList();
            return new GenomeDualSorter(
                id: Guid.NewGuid(),
                chromA: ch,
                chromB: ch,
                choices: 0u.CountUp(stageCount)
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
            var mutantIndex = randy.NextUint(genomeDualSorter.StageCount);
            var newChromA = genomeDualSorter.ChromosomeA;
            var newChromB = genomeDualSorter.ChromosomeB;

            var newChoices = genomeDualSorter.Choices;

            if (randy.NextBool(0.5))
            {
                var mutantStage = randy.RewireSorterStage(genomeDualSorter.ChromoA(mutantIndex));
                newChromA = newChromA.ReplaceAtIndex(mutantIndex, mutantStage);
            }
            else
            {
                var mutantStage = randy.RewireSorterStage(genomeDualSorter.ChromoB(mutantIndex));
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
            for (uint i = 0; i < count; i++)
            {
                yield return 0u.CountUp(genomeDualSorter.StageCount)
                    .Select(d => genomeDualSorter.Choice(d)
                        ? genomeDualSorter.ChromoA(d)
                        : genomeDualSorter.ChromoB(d))
                    .MakeSorter(id: Guid.NewGuid(), genomeId: genomeDualSorter.Id);
            }
        }

    }
}