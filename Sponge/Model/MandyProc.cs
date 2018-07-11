using System.Drawing;
using System.Windows.Media.Imaging;
using Hybridizer.Runtime.CUDAImports;
using Utils;

namespace Sponge.Model
{
    class MandyProc
    {
        const int maxiter = 5120;
        const int N = 4096;
        const float fromX = -1.40f;
        const float fromY = -0.015f;
        const float size = 0.02f;
        const float h = size / (float)N;

        [Kernel]
        public static int IterCount(float cx, float cy)
        {
            int result = 0;
            float x = 0.0f;
            float y = 0.0f;
            float xx = 0.0f, yy = 0.0f;
            while (xx + yy <= 4.0f && result < maxiter)
            {
                xx = x * x;
                yy = y * y;
                float xtmp = xx - yy + cx;
                y = 2.0f * x * y + cy;
                x = xtmp;
                result++;
            }
            return result;
        }

        [EntryPoint("run")]
        public static void MandyKernel(int[] light, int lineFrom, int lineTo)
        {
            for (int line = lineFrom + threadIdx.y + blockDim.y * blockIdx.y; line < lineTo; line += gridDim.y * blockDim.y)
            {
                for (int j = threadIdx.x + blockIdx.x * blockDim.x; j < N; j += blockDim.x * gridDim.x)
                {
                    float x = fromX + line * h;
                    float y = fromY + j * h;
                    light[line * N + j] = IterCount(x, y);
                }
            }
        }

        [EntryPoint("run2")]
        public static void MandyKernel2(int[] light, int lineFrom, int lineTo)
        {
            for (int line = lineFrom + threadIdx.y + blockDim.y * blockIdx.y; line < lineTo; line += gridDim.y * blockDim.y)
            {
                for (int j = threadIdx.x + blockIdx.x * blockDim.x; j < N; j += blockDim.x * gridDim.x)
                {
                    float x = fromX - 0.1f + line * h;
                    float y = fromY + j * h;
                    light[line * N + j] = IterCount(x, y);
                }
            }
        }


        private static dynamic wrapper;

        public static void MakeMandy(int[] light)
        {
            wrapper.MandyKernel(light, 0, N);
        }

        public static void MakeMandy2(int[] light)
        {
            wrapper.MandyKernel2(light, 0, N);
        }

        public static BitmapImage MakeTestImage()
        {
            const int redo = 1;
            var lightCuda = new int[N * N];

            var runner = HybRunner.Cuda("Wpf_H_CUDA.dll").SetDistrib(32, 32, 16, 16, 1, 0);
            wrapper = runner.Wrap(new MandyProc());

            // profile with nsight to get performance
            for (var i = 0; i < redo; ++i)
            {
                MakeMandy(lightCuda);
            }
            
            var colors = new Color[maxiter + 1];

            for (var k = 0; k < maxiter; ++k)
            {
                var red = (int) (127.0F * (float)k / (float)maxiter);
                var green = (int)(200.0F * (float)k / (float)maxiter);
                var blue = (int)(90.0F * (float)k / (float)maxiter);
                colors[k] = Color.FromArgb((byte) red, (byte) green, (byte) blue, 255);
            }
            colors[maxiter] = Color.Black;

            var bitmap = new Bitmap(N, N);
            for (var i = 0; i < N; ++i)
            {
                for (var j = 0; j < N; ++j)
                {
                    var index = i * N + j;
                    bitmap.SetPixel(i, j, colors[lightCuda[index]]);
                }
            }

            var bis = bitmap.ToImageSource();
            bis.Freeze();
            return bis;
        }

    }
}

