using System.Collections.Generic;
using System.Linq;

namespace Utils
{
    public class SbScratchPad
    {
        public SbScratchPad(IEnumerable<SbSpItem> overlaps, uint order)
        {
            Overlaps = overlaps.ToArray();
            Order = order;
        }

        private SbSpItem[] Overlaps { get; }

        private SbSpItem Overlap(uint row, uint col)
        {
            return Overlaps[row * Order + col];
        }

        public uint Order { get; }

        // search the lower triangle for the best item in col
        public SbSpItem BestOnCol(uint col)
        {
            return col.CountUp(Order)
                .Select(i => Overlap(i, col))
                .Where(o => !o.IsUsed && (o.Row != col))
                .Aggregate((i1, i2) => i1.Overlap > i2.Overlap ? i1 : i2);
        }

        public bool IsSpent()
        {
            return Order
                       .CountUp(Order)
                       .Select(i => Overlap(i, i))
                       .Count(o => !o.IsUsed) < 2;
        }

        public SbSpItem BestOnDiag()
        {
            return Order.CountUp(Order)
                .Select(i => Overlap(i, i))
                .Where(o => !o.IsUsed)
                .Aggregate((i1, i2) => i1.Overlap > i2.Overlap ? i1 : i2);
        }

        public void Mark(SbSpItem sbSp)
        {
            Overlap(sbSp.Row, sbSp.Col).IsUsed = true;
            Overlap(sbSp.Col, sbSp.Row).IsUsed = true;
            Overlap(sbSp.Col, sbSp.Col).IsUsed = true;
            Overlap(sbSp.Row, sbSp.Row).IsUsed = true;
        }
    }

    public class SbSpItem
    {
        public SbSpItem(uint row, uint col, uint overlap)
        {
            Row = row;
            Col = col;
            Overlap = overlap;
            IsUsed = false;
        }

        public bool IsUsed { get; set; }
        public uint Row { get; }
        public uint Col { get; }
        public uint Overlap { get; }
    }

    public static class SbScratchPadExt
    {
        public static SbScratchPad ToSbScratchPad(this StageBits stageBits)
        {
            var overlaps = stageBits.Order.SquareArrayCoords()
                .Select(t =>
                    new SbSpItem(row: t.Item1,
                        col: t.Item2,
                        overlap: stageBits.Overlap(row: t.Item1, col: t.Item2)));

            var scratchPad = new SbScratchPad(
                overlaps: overlaps,
                order: stageBits.Order);

            return scratchPad;
        }

        public static SbScratchPad ToSbScratchPadA(this StageBits stageBits)
        {
            var overlaps = stageBits.Order.SquareArrayCoords()
                .Select(t =>
                    new SbSpItem(row: t.Item1,
                        col: t.Item2,
                        overlap: stageBits.Overlap(row: t.Item1, col: t.Item2)));                   

            var scratchPad = new SbScratchPad(
                overlaps: overlaps,
                order: stageBits.Order);

            return scratchPad;
        }

    }
}