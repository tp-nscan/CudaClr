using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;

namespace Utils.Genome
{
    public class GenomePoolSorterBits : IEnumerable<GenomeSorterBits>
    {
        public GenomePoolSorterBits(Guid id, IEnumerable<GenomeSorterBits> dualSorterGenomes)
        {
            Id = id;
            GenomeSorterBitses = dualSorterGenomes.ToDictionary(r => r.Id);
        }

        public Dictionary<Guid, GenomeSorterBits> GenomeSorterBitses { get; }

        public Guid Id { get; }

        public IEnumerator<GenomeSorterBits> GetEnumerator()
        {
            return GenomeSorterBitses.Values.GetEnumerator();
        }

        IEnumerator IEnumerable.GetEnumerator()
        {
            return GetEnumerator();
        }

        public GenomeSorterBits this[int index] => GenomeSorterBitses.Values.ElementAt(index);
    }

    public static class GenomePoolSorterBitsExt
    {
        public static GenomePoolSorterBits ToGenomePoolSorterBits(this IRando rando, uint order,
            uint stageCount, uint poolCount)
        {
            return 0u.CountUp(poolCount)
                .Select(i => rando.ToGenomeSorterBits2(order, stageCount))
                .ToGenomePoolSorterBits(Guid.NewGuid());
        }

        public static GenomePoolSorterBits ToGenomePoolSorterBits(
            this IEnumerable<GenomeSorterBits> genomeSorterBitses, Guid id)
        {
            return new GenomePoolSorterBits(id, genomeSorterBitses);
        }
    }

}
