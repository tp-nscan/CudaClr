using System.Collections.Generic;
using System.Linq;

namespace Utils.Genome.Utils
{
    public class SbScratchPad
    {
        public SbScratchPad(IEnumerable<SbSpItem> overlaps, uint order)
        {
            Overlaps = overlaps.ToArray();
            Order = order;
        }

        private SbSpItem[] Overlaps { get; }

        public SbSpItem Overlap(uint row, uint col)
        {
            return Overlaps[row * Order + col];
        }

        public uint Order { get; }
        
        public SbSpItem BestOnCol(uint col)
        {
            var items = 0u.CountUp(Order)
                .Select(i => Overlap(i, col)).ToArray();

            return 0u.CountUp(Order)
                .Select(i => Overlap(i, col))
                .Where(o => !o.IsUsed)
                .OrderByDescending(o => o.Overlap)
                .First();
        }

        public bool IsSpent()
        {
            return 0u
                   .CountUp(Order)
                   .Select(i => Overlap(i, i))
                   .Count(o => !o.IsUsed) < 2;
        }

        public SbSpItem BestOnDiag()
        {
            return 0u.CountUp(Order)
                .Select(i => Overlap(i, i))
                .Where(o => !o.IsUsed)
                .OrderByDescending(o => o.Overlap)
                .First();
        }

        public void Mark(SbSpItem sbSp, uint order)
        {
            for (uint i = 0; i < order; i++)
            {
                Overlap(sbSp.Row, i).IsUsed = true;
                Overlap(i, sbSp.Col).IsUsed = true;
                Overlap(sbSp.Col, i).IsUsed = true;
                Overlap(i, sbSp.Row).IsUsed = true;
            }
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

        public static SbScratchPad ToSbScratchPad2(this StageBits2 stageBits)
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

        public static string PrintUsed(this SbScratchPad sbScratchPad)
        {
            return StringFuncs.GridFormat(sbScratchPad.Order, sbScratchPad.Order,
                (r, c) => $"{sbScratchPad.Overlap(r,c).IsUsed}");
        }

        public static string PrintOverlaps(this SbScratchPad sbScratchPad)
        {
            return StringFuncs.GridFormat(sbScratchPad.Order, sbScratchPad.Order,
                (r, c) => $"{sbScratchPad.Overlap(r, c).Overlap}");
        }

    }
}