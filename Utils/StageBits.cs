using System;
using System.Collections.Generic;
using System.Linq;

namespace Utils
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

        public static StageBits ToStageBits(this IRando randy, uint order, uint mask)
        {
            return new StageBits(
                    order:order, 
                    mask: mask, 
                    bits: Enumerable.Range(0, (int)(order * (order + 1) / 2))
                        .Select(i => randy.NextUint())
                );
        }

        public static IPermutation ToPermutation(this StageBits stageBits)
        {
            var permArray = new uint[stageBits.Order];
            var sbScratchPad = stageBits.ToSbScratchPad();

            while (!sbScratchPad.IsSpent())
            {
                var dSbSp = sbScratchPad.BestOnDiag();
                var bSpSp = sbScratchPad.BestOnCol(dSbSp.Col);
                permArray[bSpSp.Col] = bSpSp.Row;
                sbScratchPad.Mark(bSpSp);
            }

            return PermutationEx.MakePermutation(permArray);
        }



    }
}
