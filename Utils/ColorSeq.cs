using System;
using System.Collections.Generic;
using System.Linq;
using System.Windows.Media;

namespace Utils
{
    public static class ColorSeq
    {
        public static IEnumerable<Color> ColorTent(int length, 
            double cR, double cG, double cB,
            double wR, double wG, double wB,
            bool wrap)
        {
            return Enumerable.Range(0, length).Select(
                i => System.Windows.Media.Color.FromArgb
                    (
                        a:255,
                        r:(byte)(TentV(Dist(cR, (double)i / length, wrap), wR) * 255),
                        g:(byte)(TentV(Dist(cG, (double)i / length, wrap), wG) * 255),
                        b:(byte)(TentV(Dist(cB, (double)i / length, wrap), wB) * 255)
                    )
                 );
        }

        public static IEnumerable<byte> Tent(int length, double center, double width, bool wrap)
        {
            return Enumerable.Range(0, length).Select(
                    i => (byte)(TentV(Dist(center, (double)i/ length, wrap), width) * 255));
        }

        static double Dist(double center, double loc, bool wrap)
        {
            var absD = Math.Abs(center - loc);
            return (wrap && (absD > 0.5)) ? (1 - absD) : absD;
        }

        static double TentV(double dist, double width)
        {
            var res = dist / width;
            return (dist >= width) ? 0 : 1.0 - res;
        }
    }
}
