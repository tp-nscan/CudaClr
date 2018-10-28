using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using Utils.Genome.Sorter;

namespace Utils.Genome
{
    public class GenomePool<T> : IEnumerable<T> where T: IGuid
    {
        public GenomePool(Guid id, IEnumerable<T> dualSorterGenomes)
        {
            Id = id;
            SorterGenomes = dualSorterGenomes.ToDictionary(r => r.Id);
        }

        public Dictionary<Guid, T> SorterGenomes { get; }

        public Guid Id { get; }

        public IEnumerator<T> GetEnumerator()
        {
            return SorterGenomes.Values.GetEnumerator();
        }

        IEnumerator IEnumerable.GetEnumerator()
        {
            return GetEnumerator();
        }

        public T this[int index] => SorterGenomes.Values.ElementAt(index);

    }


    public static class GenomePoolExt
    {
        public static GenomePool<GenomeSorterDual> ToGenomePoolDualSorter(
            this IRando rando, uint order, uint stageCount, uint poolCount)
        {
            return new GenomePool<GenomeSorterDual>(Guid.NewGuid(),
                0u.CountUp(poolCount).Select(i =>
                    rando.ToDualSorterGenome(order, stageCount)));
        }

        public static GenomePool<GenomeSorterDual> ToGenomePoolDualSorter(this 
            IEnumerable<GenomeSorterDual> genomeDualSorters, Guid id)
        {
            return new GenomePool<GenomeSorterDual>(id, genomeDualSorters);
        }

        public static GenomePool<GenomeSorterBits> ToGenomePoolSorterBits(
            this IRando rando, uint order,
            uint stageCount, uint poolCount)
        {
            return 0u.CountUp(poolCount)
                .Select(i => rando.ToGenomeSorterBits2(order, stageCount))
                .ToGenomePoolSorterBits(Guid.NewGuid());
        }

        public static GenomePool<GenomeSorterBits> ToGenomePoolSorterBits(
            this IEnumerable<GenomeSorterBits> genomeSorterBitses, Guid id)
        {
            return new GenomePool<GenomeSorterBits>(id, genomeSorterBitses);
        }

        public static GenomePool<GenomeSorterStageDimer> ToGenomePoolStageDimer(
            this IRando rando, uint order, uint stageCount, uint poolCount)
        {
            return 0u.CountUp(poolCount)
                .Select(i => rando.ToGenomeDimer(order, stageCount))
                .ToGenomePoolStageDimer(Guid.NewGuid());
        }

        public static GenomePool<GenomeSorterStageDimer> ToGenomePoolStageDimer(
            this IEnumerable<GenomeSorterStageDimer> genomeDimers, Guid id)
        {
            return new GenomePool<GenomeSorterStageDimer>(id, genomeDimers);
        }

        public static GenomePool<GenomeSorterStageDimer> ToRecombCoarse(
            this IEnumerable<GenomeSorterStageDimer> genomeDimers, IRando rando)
        {
            return genomeDimers.ToRandomPairs(rando)
                    .SelectMany(rp => rando.RecombineCoarse(rp.Item1, rp.Item2).Split())
                    .ToGenomePoolStageDimer(Guid.NewGuid());
        }

        public static GenomePool<GenomeSorterStageDimer> ToRecombFine(
            this IEnumerable<GenomeSorterStageDimer> genomeDimers, IRando rando)
        {
            return genomeDimers.ToRandomPairs(rando)
                .SelectMany(rp => rando.RecombineFine(rp.Item1, rp.Item2).Split())
                .ToGenomePoolStageDimer(Guid.NewGuid());
        }

        public static GenomePool<GenomeSorterConjOrbit> ToGenomePoolConjOrbits(
            this IRando rando, uint order, uint stageCount, uint poolCount)
        {
            return 0u.CountUp(poolCount)
                .Select(i => rando.ToGenomeConjOrbit(order, stageCount))
                .ToGenomePoolConjOrbits(Guid.NewGuid());
        }

        public static GenomePool<GenomeSorterConjOrbit> ToGenomePoolConjOrbits(
            this IEnumerable<GenomeSorterConjOrbit> genomeConjOrbits, Guid id)
        {
            return new GenomePool<GenomeSorterConjOrbit>(id, genomeConjOrbits);
        }
    }
}
