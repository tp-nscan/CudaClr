using System;
using System.Drawing;

namespace Utils
{
    public static class LegendExt
    {
        static LegendExt()
        {
            TriColors = new Color[256];
            GrayScales = new Color[256];
            const double step = Math.PI/(128.0 * 3.0);
            const double halfR = 127.5;

            for (double i = 0; i < 256; i++)
            {
                TriColors[(int) i] = Color.FromArgb(
                    alpha: 255,
                    red: (byte)(halfR + halfR * Math.Sin(i * step)),
                    green: (byte)(halfR + halfR * Math.Sin((i + 85.33) * step)),
                    blue: (byte)(halfR + halfR * Math.Sin((i + 170.66) * step))
                    );

                GrayScales[(int) i] = Color.FromArgb(255, 
                    (byte)i, 
                    (byte)i, 
                    (byte)i);

            }
        }

        private static readonly Color[] TriColors;
        private static readonly Color[] GrayScales;

        // Maps [0, 255] to grey scale
        public static Color[] ToGrayScale(this int[] data, int length)
        {
            var colors = new Color[length];
            for (var k = 0; k < length; ++k)
            {
                colors[k] = Color.FromArgb(255, (byte)data[k], (byte)data[k], (byte)data[k]);
            }
            return colors;
        }

        public static Color[] ToGrayScale2(this int[] data, int length)
        {
            var colors = new Color[length];
            for (var k = 0; k < length; ++k)
            {
                colors[k] = GrayScales[(data[k] + 4096) % 256];
            }
            return colors;
        }


        // Maps [0, 255] to 3 color gradient
        public static Color[] ToTriColors(this int[] data, int length)
        {
            var colors = new Color[length];
            for (var k = 0; k < length; ++k)
            {
                colors[k] = TriColors[(data[k] + 4096) % 256];
            }
            return colors;
        }

    }
}