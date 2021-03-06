﻿using System;
using System.Linq;
using Utils.Sorter;

namespace Utils.Genome.Sorter
{
    public class GenomeSorterConjOrbit : IGuid
    {
        public GenomeSorterConjOrbit(Guid id, IPermutation twoCycle, IPermutation conj, uint order, uint stageCount)
        {
            Id = id;
            TwoCycle = twoCycle;
            Conj = conj;
            Order = order;
            Stagecount = stageCount;
        }

        public Guid Id { get; }

        public IPermutation TwoCycle { get; }

        public IPermutation Conj { get; }

        public uint Order { get; }

        public uint Stagecount { get; }
        
    }

    public static class GenomeSorterConjOrbitExt
    {
        public static GenomeSorterConjOrbit ToGenomeConjOrbit(this IRando randy, uint order, uint stageCount)
        {
            return new GenomeSorterConjOrbit(
                id: Guid.NewGuid(),
                twoCycle: randy.ToFullTwoCyclePermutation(order),
                conj: randy.ToPermutation(order),
                order: order,
                stageCount: stageCount);
        }

        public static GenomeSorterConjOrbit Mutate(this GenomeSorterConjOrbit genomeConjOrbit, IRando randy)
        {
            var newTs = genomeConjOrbit.TwoCycle;
            var newPerm = genomeConjOrbit.Conj;

            if (randy.NextBool(0.5))
            {
                newTs = newTs.ConjugateByRandomSingleTwoCycle(randy);
            }
            else
            {
                newPerm = newPerm.ConjugateByRandomPermutation(randy);
            }

            return new GenomeSorterConjOrbit(
                id: Guid.NewGuid(),
                twoCycle: newTs,
                conj: newPerm,
                order: genomeConjOrbit.Order,
                stageCount: genomeConjOrbit.Stagecount);
        }

        public static ISorter ToSorter(this GenomeSorterConjOrbit genomeConjOrbit, uint maxOrbitSize)
        {
            var orbs = genomeConjOrbit.TwoCycle
                                     .GetConjOrbit(genomeConjOrbit.Conj, maxOrbitSize)
                                     .Keys.ToRoundRobin().Take((int)maxOrbitSize)
                                     .Select(p=>p.ToSorterStage(0u));

            return new global::Utils.Sorter.Sorter(
                id: Guid.NewGuid(),
                genomeId: genomeConjOrbit.Id,
                stages: orbs);
        }

    }
}
