using System;
using System.Collections.Generic;
using System.Linq;
using Utils.Genome.Utils;
using Utils.Sorter;

namespace Utils.Genome
{
    //public class GenomeSorterBits
    //{
    //    public GenomeSorterBits(Guid id, IEnumerable<StageBits> stageBits)
    //    {
    //        Id = id;
    //        StageBits = stageBits.ToArray();

    //    }

    //    public Guid Id { get; }

    //    public StageBits[] StageBits { get; }

    //    public uint Order => StageBits[0].Order;

    //    public uint StageBitsCount => (uint) StageBits.Length;
    //}

    public class GenomeSorterBits
    {
        public GenomeSorterBits(Guid id, IEnumerable<StageBits2> stageBits)
        {
            Id = id;
            StageBits = stageBits.ToArray();

        }

        public Guid Id { get; }

        public StageBits2[] StageBits { get; }

        public uint Order => StageBits[0].Order;

        public uint StageBitsCount => (uint)StageBits.Length;
    }

    public static class GenomeSorterBitsExt
    {

        public static GenomeSorterBits ToGenomeSorterBits2(this IRando randy, uint order, uint stageCount)
        {
            return 0u.CountUp(stageCount)
                .Select(i => randy.ToStageBits2(order))
                .ToGenomeSorterBits2(Guid.NewGuid());
        }

        //public static GenomeSorterBits ToGenomeSorterBits(this IRando randy, uint order, uint stageCount)
        //{
        //    return 0u.CountUp(stageCount)
        //        .Select(i => randy.ToStageBits(order))
        //        .ToGenomeSorterBits(Guid.NewGuid());
        //}


        public static GenomeSorterBits ToGenomeSorterBits2(this IEnumerable<StageBits2> stageBits, Guid id)
        {
            return new GenomeSorterBits(id: id, stageBits: stageBits);
        }

        //public static GenomeSorterBits ToGenomeSorterBits(this IEnumerable<StageBits> stageBits, Guid id)
        //{
        //    return new GenomeSorterBits(id: id, stageBits: stageBits);
        //}

        public static GenomeSorterBits Mutate(this GenomeSorterBits sorterBits, Guid id, IRando randy,
            double mutationRate)
        {
            return new GenomeSorterBits(id: id,
                stageBits: sorterBits.StageBits.Select(sb => sb.Mutate(randy, mutationRate)));
        }

        public static ISorter MakePhenotype(this GenomeSorterBits genomeSorterBits,
            IRando randy)
        {
            return 0u.CountUp(genomeSorterBits.StageBitsCount)
                .Select(i=> genomeSorterBits.StageBits[i].ToSorterStage(i))
                .MakeSorter(id: Guid.NewGuid(), genomeId: genomeSorterBits.Id);
        }
    }
}
