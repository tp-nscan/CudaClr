using System;
using System.Linq;

namespace CublasClr
{
    public static class GpuMatrixUtils
    {
        public static T[] RowMajorToColMajor<T>(T[] row_maj, int rows, int cols)
        {
            T[] colMaj = new T[rows*cols];

            for(var i=0; i<rows; i++)
            {
                for (var j = 0; j < cols; j++)
                {
                    colMaj[j* rows + i] = row_maj[i * cols + j];
                }
            }
            return colMaj;
        }

        public static float[] AA()
        {
            return new float[] {1, 0,
                                0, 2,
                                3, 0};
        }

        public static float[] BB()
        {
            return new float[] {1, 0, 0,
                                0, 0.2f, 0.3f};
        }

        public static float[] MakeIdentity(uint rows, uint cols)
        {
            var data = new float[rows * cols];
            var ext = Math.Min(rows, cols);
            for (var i = 0; i < ext; i++)
            {
                data[i * rows + i] = 1.0f;
            }
            return data;
        }

        public static float[] MakeZeroes(uint rows, uint cols)
        {
            return new float[rows * cols];
        }

        public static float[] MakeIdentiPoke(uint rows, uint cols)
        {
            var data = new float[rows * cols];
            var ext = Math.Min(rows, cols);
            for (var i = 0; i < ext; i++)
            {
                data[i * rows + i] = 1.0f;
            }
            data[rows] = 1.0f;
            return data;
        }

        public static float[] MakeColumnMajor(uint rows, uint cols)
        {
            return Enumerable.Range(0, (int)(rows * cols))
                                  .Select(i => (float)i)
                                  .ToArray();
        }

        public static void MatrixMult(float[] C, float[] A, float[] B, uint hA, uint wA, uint wB)
        {
            for (var i = 0; i < hA; ++i)
                for (var j = 0; j < wB; ++j)
                {
                    double sum = 0;
                    for (var k = 0; k < wA; ++k)
                    {
                        double a = A[i * wA + k];
                        double b = B[k * wB + j];
                        sum += a * b;
                    }

                    C[i * wB + j] = (float)sum;
                }
        }



        public static float[] UniformRandomFloatArray(int len)
        {
            var faRet = new float[len];
            var rando = new Random();

            for (var i = 0; i < len; i++)
            {
                faRet[i] = (float)rando.NextDouble();
            }

            return faRet;
        }

    }

}
