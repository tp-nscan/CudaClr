using System;
using System.Collections.Generic;
using System.Linq;
using Utils.Genome.Utils;
using Utils.Sorter;

namespace Utils.Genome.Sorter
{
    public class GenomeSorterStageDimer : IGuid
    {
        public GenomeSorterStageDimer(Guid id, IEnumerable<StageDimer> genomeStageDimers)
        {
            Id = id;
            GenomeStageDimers = genomeStageDimers.ToArray();
        }

        public Guid Id { get; }

        public StageDimer[] GenomeStageDimers { get; }

    }


    public static class GenomeSorterStageDimerExt
    {
        public static GenomeSorterStageDimer ToGenomeDimer(this IRando randy, uint order, uint stageCount)
        {
            return 0u.CountUp(stageCount / 2)
                .Select(i => randy.ToGenomeStageDimer(order: order))
                .ToGenomeDimer(Guid.NewGuid());
        }

        public static GenomeSorterStageDimer ToGenomeDimer(this IEnumerable<StageDimer> genomeStageDimers, 
            Guid id)
        {
            return new GenomeSorterStageDimer(id: id, genomeStageDimers: genomeStageDimers);
        }

        public static Tuple<GenomeSorterStageDimer, GenomeSorterStageDimer> RecombineCoarse(this IRando randy, GenomeSorterStageDimer gdA, GenomeSorterStageDimer gdB)
        {
            var aList = gdA.GenomeStageDimers.ToList();
            var bList = gdB.GenomeStageDimers.ToList();
            //var combies = randy.Recombo(aList, bList);
            var combies = aList.Recombo(bList, randy.NextUint((uint)bList.Count()));
            return new Tuple<GenomeSorterStageDimer, GenomeSorterStageDimer>(
                combies.Item1.ToGenomeDimer(Guid.NewGuid()),
                combies.Item2.ToGenomeDimer(Guid.NewGuid()));
        }

        public static Tuple<GenomeSorterStageDimer, GenomeSorterStageDimer> RecombineFine(this IRando randy, GenomeSorterStageDimer gdA, GenomeSorterStageDimer gdB)
        {
            var al = gdA.GenomeStageDimers.SelectMany(sd => sd.ToPermutations()).ToList();
            var aList = gdA.GenomeStageDimers.ToList();
            var bList = gdB.GenomeStageDimers.ToList();
           // var combies = aList.Recombo(bList, randy.NextUint((uint)bList.Count()));
            var combies = aList.RecomboL2(bList, 
                randy.NextUint((uint)bList.Count()), 
                StageDimerExt.RecomboP(randy.NextUint(4u)));
            return new Tuple<GenomeSorterStageDimer, GenomeSorterStageDimer>(
                combies.Item1.ToGenomeDimer(Guid.NewGuid()),
                combies.Item1.ToGenomeDimer(Guid.NewGuid()));
        }

        public static ISorter ToSorter(this GenomeSorterStageDimer genomeDimer)
        {
            var stages = genomeDimer.GenomeStageDimers
                .SelectMany(gsd => gsd.ToPhenotype())
                .Select(p => p.ToSorterStage(0));

            return stages.ToSorter(Guid.NewGuid(), genomeDimer.Id);
        }

        public static GenomeSorterStageDimer Mutate(this GenomeSorterStageDimer genomeDimer, IRando rando)
        {
            var mutantIndex = rando.NextInt(genomeDimer.GenomeStageDimers.Length);
            var gsdToReplace = genomeDimer.GenomeStageDimers[mutantIndex];
            StageDimer gsdMutant = gsdToReplace.Mutate(rando);
            return genomeDimer.GenomeStageDimers
                              .ReplaceAtIndex((uint)mutantIndex, gsdMutant)
                              .ToGenomeDimer(Guid.NewGuid());
        }
    }

}
