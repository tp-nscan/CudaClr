using System.Collections.Generic;
using System.Linq;
using Utils.Sorter;

namespace Utils.Genome.Utils
{
    public class StageBits
    {
        public StageBits(uint order, uint mask, IEnumerable<uint> bits)
        {
            Mask = mask;
            Bits = bits.ToArray();
            Order = order;
            _maskOverlaps = Bits.BitOverlaps(Mask).ToArray();
        }

        public uint[] Bits { get; }

        public uint Mask { get; }

        public uint Order { get; }

        private readonly uint[] _maskOverlaps;

        public uint Overlap(uint row, uint col)
        {
            return _maskOverlaps[EnumerableExt.LowerTriangularIndex(row, col)];
        }
    }


    public static class StageBitsExt
    {

        public static StageBits ToStageBits(this IRando randy, uint order)
        {
            return 0u.CountUp(order * (order + 1) / 2)
                     .Select(i => randy.NextUint())
                     .ToStageBits(order:order, mask: randy.NextUint());
        }


        public static StageBits ToStageBits(this IEnumerable<uint> bits, uint order, uint mask)
        {
            return new StageBits(
                order: order,
                mask: mask,
                bits: bits
            );
        }

        public static StageBits Mutate(this StageBits stageBits, IRando randy, double mutationRate)
        {
            return new StageBits(
                    order:stageBits.Order,
                    mask: stageBits.Mask.AsEnumerable().Mutate(randy, mutationRate).First(),
                    bits: stageBits.Bits.Mutate(randy, mutationRate)
                );
        }

        public static ISorterStage ToSorterStage(this StageBits stageBits, uint stageNumber)
        {
            return stageBits.ToPermutation().ToSorterStage(stageNumber);
        }

        public static IPermutation ToPermutation(this StageBits stageBits)
        {
            var permArray = 0u.CountUp(stageBits.Order).ToArray();
            var sbScratchPad = stageBits.ToSbScratchPad();

            //System.Diagnostics.Debug.WriteLine(sbScratchPad.PrintOverlaps());
            //System.Diagnostics.Debug.WriteLine(sbScratchPad.PrintUsed());

            while (!sbScratchPad.IsSpent())
            {
                var dSbSp = sbScratchPad.BestOnDiag();
                dSbSp.IsUsed = true;
                var bSpSp = sbScratchPad.BestOnCol(dSbSp.Col);
                permArray[bSpSp.Col] = bSpSp.Row;
                permArray[bSpSp.Row] = bSpSp.Col;
                sbScratchPad.Mark(bSpSp, stageBits.Order);

              //  System.Diagnostics.Debug.WriteLine(sbScratchPad.PrintUsed());
            }

            return PermutationEx.MakePermutation(permArray);
        }


    }
}
