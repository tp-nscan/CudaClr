using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Utils.Sorter;

namespace Utils.Genome
{
    public class GenomeDimer : IGuid
    {
        public GenomeDimer(Guid id, IEnumerable<GenomeStageDimer> genomeStageDimers)
        {
            Id = id;
            GenomeStageDimers = genomeStageDimers.ToArray();
        }

        public Guid Id { get; }

        public GenomeStageDimer[] GenomeStageDimers { get; }

    }


    public static class GenomeDimerExt
    {
        public static GenomeDimer ToGenomeDimer(this IRando randy, uint order, uint stageCount)
        {
            return 0u.CountUp(stageCount / 2)
                .Select(i => randy.ToGenomeStageDimer(order: order))
                .ToGenomeDimer(Guid.NewGuid());
        }

        public static GenomeDimer ToGenomeDimer(this IEnumerable<GenomeStageDimer> genomeStageDimers, 
            Guid id)
        {
            return new GenomeDimer(id: id, genomeStageDimers: genomeStageDimers);
        }

        public static ISorter ToSorter(this GenomeDimer genomeDimer)
        {
            var stages = genomeDimer.GenomeStageDimers
                .SelectMany(gsd => gsd.ToPhenotype())
                .Select(p => p.ToSorterStage(0));

            return stages.ToSorter(Guid.NewGuid(), genomeDimer.Id);
        }

        public static GenomeDimer Mutate(this GenomeDimer genomeDimer, IRando rando)
        {
            var mutantIndex = rando.NextInt(genomeDimer.GenomeStageDimers.Length);
            var gsdToReplace = genomeDimer.GenomeStageDimers[mutantIndex];
            GenomeStageDimer gsdMutant = gsdToReplace.Mutate(rando);
            return genomeDimer.GenomeStageDimers
                              .ReplaceAtIndex((uint)mutantIndex, gsdMutant)
                              .ToGenomeDimer(Guid.NewGuid());
        }
    }

}
