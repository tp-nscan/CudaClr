using System;
using System.Collections.Generic;

namespace Utils.Genome.Utils
{
    public class StageDimer
    {
        public StageDimer(IPermutation stage1, IPermutation stage2, IPermutation modifier)
        {
            Stage1 = stage1;
            Stage2 = stage2;
            Modifier = modifier;
        }

        public IPermutation Stage1 { get; }

        public IPermutation Stage2 { get; }

        public IPermutation Modifier { get; }
    }

    public static class StageDimerExt
    {
        public static StageDimer ToGenomeStageDimer(this IRando randy, uint order)
        {
            return new StageDimer(
                    stage1: randy.ToFullTwoCyclePermutation(order), 
                    stage2: randy.ToFullTwoCyclePermutation(order), 
                    modifier: randy.ToPermutation(order)
                );
        }

        public static StageDimer ToGenomeStageDimer(this IPermutation[] perms)
        {
            return new StageDimer(
                stage1: perms[0],
                stage2: perms[1],
                modifier: perms[2]
            );
        }

        public static IEnumerable<IPermutation> ToPermutations(this StageDimer genomeStageDimer)
        {
            yield return genomeStageDimer.Stage1;
            yield return genomeStageDimer.Stage2;
            yield return genomeStageDimer.Modifier;
        }

        public static StageDimer Mutate(this StageDimer genomeStageDimer, IRando randy)
        {
            var newStage1 = genomeStageDimer.Stage1;
            var newStage2 = genomeStageDimer.Stage2;
            var newModifier = genomeStageDimer.Modifier;

            var spot = randy.NextUint(3);
            switch (spot)
            {
                case 0:
                    newStage1 = newStage1.ConjugateBy(randy.ToSingleTwoCyclePermutation(newStage1.Order));
                    break;
                case 1:
                    newStage2 = newStage2.ConjugateBy(randy.ToSingleTwoCyclePermutation(newStage2.Order));
                    break;
                case 2:
                    newModifier = newModifier.ConjugateBy(randy.ToSingleTwoCyclePermutation(newModifier.Order));
                    break;
                default:
                    throw new Exception($"spot {spot} not handled in StageDimerExt.Mutate");
            }

            return new StageDimer(
                stage1: newStage1,
                stage2: newStage2,
                modifier: newModifier
            );
        }

        public static IEnumerable<IPermutation> ToPhenotype(this StageDimer genomeStageDimer)
        {
            //yield return genomeStageDimer.Stage1;
            //yield return genomeStageDimer.Stage2;

            yield return genomeStageDimer.Stage1.ConjugateBy(genomeStageDimer.Modifier);
            yield return genomeStageDimer.Stage2.ConjugateBy(genomeStageDimer.Modifier);
        }

        public static Tuple<StageDimer, StageDimer> Recombo(this StageDimer gsdA, 
            StageDimer gsdB, uint pos)
        {
            switch (pos)
            {
                case 0:
                    return new Tuple<StageDimer, StageDimer>(gsdA, gsdB);
                case 1:
                    return new Tuple<StageDimer, StageDimer>(
                        new StageDimer(stage1:gsdA.Stage1, stage2:gsdB.Stage2, modifier:gsdB.Modifier),
                        new StageDimer(stage1: gsdB.Stage1, stage2: gsdA.Stage2, modifier: gsdA.Modifier));
                case 2:
                    return new Tuple<StageDimer, StageDimer>(
                        new StageDimer(stage1: gsdA.Stage1, stage2: gsdA.Stage2, modifier: gsdB.Modifier),
                        new StageDimer(stage1: gsdB.Stage1, stage2: gsdB.Stage2, modifier: gsdA.Modifier));
                case 3:
                    return new Tuple<StageDimer, StageDimer>(gsdB, gsdA);
                default:
                    throw new Exception($"pos {pos} is out of bounds in Recombo");
            }
        }

        public static Func<StageDimer, StageDimer, Tuple<StageDimer, StageDimer>>
            RecomboP(uint pos)
        {
            return (gsdA, gsdB) => gsdA.Recombo(gsdB, pos);
        }
    }
}
