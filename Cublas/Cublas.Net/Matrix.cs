using System;
using System.Collections.Immutable;
using System.Linq;

namespace Cublas.Net
{
    public enum MatrixFormat
    {
        Column_Major, // the one CUBLAS likes
        Row_Major
    }

    public class Matrix<T>
    {
        public Matrix(
                    uint _rows, uint _cols,
                    ImmutableArray<T> host_data,
                    MatrixFormat matrixFormat)
        {
            Rows = _rows;
            Cols = _cols;
            Data = host_data;
            MatrixFormat = matrixFormat;
        }

        public uint Cols { get; }
        public uint Rows { get; }
        public ImmutableArray<T> Data { get; }
        public MatrixFormat MatrixFormat { get; }
    }

    public static class MatrixUtils
    {

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



        public static Matrix<float> RandomFloatMatrix(uint rows, uint cols, MatrixFormat matrixFormat)
        {
            return new Matrix<float>(
                _rows: rows,
                _cols: cols,
                host_data: ImmutableArray.Create(UniformRandomFloatArray(rows*cols)),
                matrixFormat: matrixFormat);
        }

        public static Matrix<T> ToRowMajor<T>(this Matrix<T> colMajor)
        {
            if(colMajor.MatrixFormat == MatrixFormat.Row_Major)
            {
                return colMajor;
            }

            return new Matrix<T>(
                _rows: colMajor.Rows,
                _cols: colMajor.Cols,
                host_data: ImmutableArray.Create(colMajor.Data.ToArray().ToRowMajor(rows:colMajor.Rows, cols:colMajor.Cols)),
                matrixFormat: MatrixFormat.Row_Major);
        }

        public static Matrix<T> ToColMajor<T>(this Matrix<T> rowMajor)
        {
            if (rowMajor.MatrixFormat == MatrixFormat.Column_Major)
            {
                return rowMajor;
            }

            return new Matrix<T>(
                _rows: rowMajor.Rows,
                _cols: rowMajor.Cols,
                host_data: ImmutableArray.Create(rowMajor.Data.ToArray().ToColMajor(rows: rowMajor.Rows, cols: rowMajor.Cols)),
                matrixFormat: MatrixFormat.Column_Major);
        }

        public static T[] ToColMajor<T>(this T[] row_maj, uint rows, uint cols)
        {
            T[] colMaj = new T[rows * cols];

            for (var i = 0; i < rows; i++)
            {
                for (var j = 0; j < cols; j++)
                {
                    colMaj[j * rows + i] = row_maj[i * cols + j];
                }
            }
            return colMaj;
        }

        public static T[] ToRowMajor<T>(this T[] col_maj, uint rows, uint cols)
        {
            T[] rowMaj = new T[rows * cols];

            for (var i = 0; i < rows; i++)
            {
                for (var j = 0; j < cols; j++)
                {
                    rowMaj[i * rows + j] = col_maj[j * cols + i];
                }
            }
            return rowMaj;
        }

        public static void RowMajorMatrixMult(float[] C, float[] A, float[] B, uint hA, uint wA, uint wB)
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

        public static float[] UniformRandomFloatArray(uint len)
        {
            var faRet = new float[len];
            var rando = new Random();

            for (var i = 0; i < len; i++)
            {
                faRet[i] = (float)rando.NextDouble();
            }

            return faRet;
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
    }
}
