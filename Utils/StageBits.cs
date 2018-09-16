using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Utils
{
    public class StageBits
    {
        public StageBits(uint order, uint mask, uint[] bits)
        {
            Mask = mask;
            Bits = bits;
            Order = order;
            _maskOverlaps = bits.BitOverlaps(Mask).ToArray();
        }

        public uint[] Bits { get; }

        public uint Mask { get; }
        public uint Order { get; }

        private readonly uint[] _maskOverlaps;

        public uint Overlap(uint row, uint col)
        {
            return (row > col) ? _maskOverlaps[row * Order + col] : 
                                 _maskOverlaps[col * Order + row];
        }

    }

    public class SbScratchPad
    {
        public SbScratchPad(IEnumerable<SbSpItem> overlaps, uint order)
        {
            Overlaps = overlaps.ToArray();
            Order = order;
        }

        private SbSpItem[] Overlaps { get; }

        private SbSpItem Overlap(int row, int col)
        {
            return Overlaps[row * Order + col];
        }

        public uint Order { get; }

        public SbSpItem BestRow(int col)
        {
            return Enumerable.Range(0, (int) Order)
                .Select(i => Overlap(i, col))
                .Where(o => !o.IsUsed)
                .Aggregate((i1, i2) => i1.Overlap > i2.Overlap ? i1 : i2);
        }

        public void Mark(SbSpItem sbSp)
        {
            Overlap(sbSp.Row, sbSp.Col).IsUsed = true;
            Overlap(sbSp.Col, sbSp.Row).IsUsed = true;
        }
    }

    public class SbSpItem
    {
        public SbSpItem(int row, int col, int overlap)
        {
            Row = row;
            Col = col;
            Overlap = overlap;
            IsUsed = (row != col);
        }

        public bool IsUsed { get; set; }
        public int Row { get; }
        public int Col { get; }
        public int Overlap { get; }
    }

    public static class StageBitsExt
    {

        public static StageBits ToStageBits(this IRando randy, uint order)
        {
            return new StageBits(order:order, mask:1, bits:new uint[]{});
        }

        public static IPermutation ToPermutation(this StageBits stageBits)
        {
            var scratchPad = new SbScratchPad(
                overlaps: Enumerable.Empty<SbSpItem>(),
                order:stageBits.Order);
            return PermutationEx.MakePermutation(new int[]{});
        }



    }
}
