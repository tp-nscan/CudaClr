using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;

namespace Utils
{
    public class GenomePoolDualSorter : IEnumerable<GenomeDualSorter>
    {
        public GenomePoolDualSorter(Guid id, IEnumerable<GenomeDualSorter> dualSorterGenomes)
        {
            Id = id;
            DualSorterGenomes = dualSorterGenomes.ToDictionary(r => r.Id);
        }

        public Dictionary<Guid, GenomeDualSorter> DualSorterGenomes { get; }

        public Guid Id { get; }

        public IEnumerator<GenomeDualSorter> GetEnumerator()
        {
            return DualSorterGenomes.Values.GetEnumerator();
        }

        IEnumerator IEnumerable.GetEnumerator()
        {
            return GetEnumerator();
        }

        public GenomeDualSorter this[int index] => DualSorterGenomes.Values.ElementAt(index);

    }


    public static class GenomePoolDualSorterExt
    {
        public static GenomePoolDualSorter ToGenomePoolDualSorter(this IRando rando, uint order, 
            uint stageCount, uint poolCount)
        {
            return new GenomePoolDualSorter(Guid.NewGuid(),
                    0u.CountUp(poolCount).Select(i =>
                    rando.ToDualSorterGenome(order, stageCount)));
        }
    }
}
