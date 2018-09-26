using System.Collections.Generic;
using System.Linq;
using Utils.Sorter;

namespace Utils.Genome.Utils
{
    public class StageBits2
    {
        public StageBits2(uint order, IEnumerable<uint> masks, IEnumerable<uint> bits)
        {
            Masks = masks.ToArray();
            Bits = bits.ToArray();
            Order = order;
            _maskOverlaps = 0u.CountUp(order * (order + 1) / 2)
                .Select(i => Bits[i].BitOverlap(Masks[i / order]))
                .ToArray();
        }

        public uint[] Bits { get; }

        public uint[] Masks { get; }

        public uint Order { get; }

        private readonly uint[] _maskOverlaps;

        public uint Overlap(uint row, uint col)
        {
            return _maskOverlaps[EnumerableExt.LowerTriangularIndex(row, col)];
        }
    }


    public static class StageBits2Ext
    {

        public static StageBits2 ToStageBits2(this IRando randy, uint order)
        {
            return 0u.CountUp(order * (order + 1) / 2)
                     .Select(i => randy.NextUint())
                     .ToStageBits2(order: order, 
                                   masks: 0u.CountUp(order)
                                       .Select(i => randy.NextUint())
                    );
        }


        public static StageBits2 ToStageBits2(this IEnumerable<uint> bits, uint order, 
            IEnumerable<uint> masks)
        {
            return new StageBits2(
                order: order,
                masks: masks,
                bits: bits
            );
        }

        public static StageBits2 Mutate(this StageBits2 stageBits, IRando randy, double mutationRate)
        {
            return new StageBits2(
                    order: stageBits.Order,
                    masks: stageBits.Masks.Mutate(randy, mutationRate),
                    bits: stageBits.Bits.Mutate(randy, mutationRate)
                );
        }

        public static ISorterStage ToSorterStage(this StageBits2 stageBits, uint stageNumber)
        {
            return stageBits.ToPermutation().ToSorterStage(stageNumber);
        }

        public static IPermutation ToPermutation(this StageBits2 stageBits)
        {
            var permArray = 0u.CountUp(stageBits.Order).ToArray();
            var sbScratchPad = stageBits.ToSbScratchPad2();

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
