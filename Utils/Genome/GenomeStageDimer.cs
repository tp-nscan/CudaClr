using System;
using System.Collections.Generic;
using System.Linq;

namespace Utils.Genome
{
    public class GenomeStageDimer
    {
        public GenomeStageDimer(IPermutation stage1, IPermutation stage2, IPermutation modifier)
        {
            Stage1 = stage1;
            Stage2 = stage2;
            Modifier = modifier;
        }

        public IPermutation Stage1 { get; }

        public IPermutation Stage2 { get; }

        public IPermutation Modifier { get; }
    }

    public static class GenomeStageDimerExt
    {
        public static GenomeStageDimer ToGenomeStageDimer(this IRando randy, uint order)
        {
            return new GenomeStageDimer(
                    stage1: randy.ToFullTwoCyclePermutation(order), 
                    stage2: randy.ToFullTwoCyclePermutation(order), 
                    modifier: randy.ToPermutation(order)
                );
        }

        public static GenomeStageDimer ToGenomeStageDimer(this IPermutation[] perms)
        {
            return new GenomeStageDimer(
                stage1: perms[0],
                stage2: perms[1],
                modifier: perms[2]
            );
        }

        public static IEnumerable<IPermutation> ToPermutations(this GenomeStageDimer genomeStageDimer)
        {
            yield return genomeStageDimer.Stage1;
            yield return genomeStageDimer.Stage2;
            yield return genomeStageDimer.Modifier;
        }

        public static GenomeStageDimer Mutate(this GenomeStageDimer genomeStageDimer, IRando randy)
        {
            var newStage1 = genomeStageDimer.Stage1;
            var newStage2 = genomeStageDimer.Stage2;
            var newModifier = genomeStageDimer.Modifier;

            var spot = randy.NextUint(3);
            switch (spot)
            {
                case 0:
                    newStage1 = newStage1.ToConjugate(randy.ToSingleTwoCyclePermutation(newStage1.Order));
                    break;
                case 1:
                    newStage2 = newStage2.ToConjugate(randy.ToSingleTwoCyclePermutation(newStage2.Order));
                    break;
                case 2:
                    newModifier = newModifier.ToConjugate(randy.ToSingleTwoCyclePermutation(newModifier.Order));
                    break;
                default:
                    throw new Exception($"spot {spot} not handled in GenomeStageDimerExt.Mutate");
            }

            return new GenomeStageDimer(
                stage1: newStage1,
                stage2: newStage2,
                modifier: newModifier
            );
        }

        public static IEnumerable<IPermutation> ToPhenotype(this GenomeStageDimer genomeStageDimer)
        {
            //yield return genomeStageDimer.Stage1;
            //yield return genomeStageDimer.Stage2;

            yield return genomeStageDimer.Stage1.ToConjugate(genomeStageDimer.Modifier);
            yield return genomeStageDimer.Stage2.ToConjugate(genomeStageDimer.Modifier);
        }

        public static Tuple<GenomeStageDimer, GenomeStageDimer> Recombo(this GenomeStageDimer gsdA, 
            GenomeStageDimer gsdB, uint pos)
        {
            switch (pos)
            {
                case 0:
                    return new Tuple<GenomeStageDimer, GenomeStageDimer>(gsdA, gsdB);
                case 1:
                    return new Tuple<GenomeStageDimer, GenomeStageDimer>(
                        new GenomeStageDimer(stage1:gsdA.Stage1, stage2:gsdB.Stage2, modifier:gsdB.Modifier),
                        new GenomeStageDimer(stage1: gsdB.Stage1, stage2: gsdA.Stage2, modifier: gsdA.Modifier));
                case 2:
                    return new Tuple<GenomeStageDimer, GenomeStageDimer>(
                        new GenomeStageDimer(stage1: gsdA.Stage1, stage2: gsdA.Stage2, modifier: gsdB.Modifier),
                        new GenomeStageDimer(stage1: gsdB.Stage1, stage2: gsdB.Stage2, modifier: gsdA.Modifier));
                case 3:
                    return new Tuple<GenomeStageDimer, GenomeStageDimer>(gsdB, gsdA);
                default:
                    throw new Exception($"pos {pos} is out of bounds in Recombo");
            }
        }

        public static Func<GenomeStageDimer, GenomeStageDimer, Tuple<GenomeStageDimer, GenomeStageDimer>>
            RecomboP(uint pos)
        {
            return (gsdA, gsdB) => gsdA.Recombo(gsdB, pos);
        }
    }
}
