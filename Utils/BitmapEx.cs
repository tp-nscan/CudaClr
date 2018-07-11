using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.Windows.Media.Imaging;

namespace Utils
{
    public static class BitmapEx
    {
        public static BitmapImage ToImageSource(this Bitmap bitmap)
        {
            using (var memory = new MemoryStream())
            {
                bitmap.Save(memory, ImageFormat.Bmp);
                memory.Position = 0;
                var bitmapimage = new BitmapImage();
                bitmapimage.BeginInit();
                bitmapimage.StreamSource = memory;
                bitmapimage.CacheOption = BitmapCacheOption.OnLoad;
                bitmapimage.EndInit();

                return bitmapimage;
            }
        }


        public static Bitmap ToBitmap(this Color[] colors, int span)
        {
            var bitmap = new Bitmap(span, span);

            for (var i = 0; i < span; ++i)
            {
                for (var j = 0; j < span; ++j)
                {
                    var index = i * span + j;
                    bitmap.SetPixel(i, j, colors[index]);
                }
            }
            return bitmap;
        }

    }
}
