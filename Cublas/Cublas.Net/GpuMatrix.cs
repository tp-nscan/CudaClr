using System;
using System.Collections.Immutable;
using System.Linq;
using CuArrayClr;
using CublasClr;

namespace Cublas.Net
{
    public enum DevHostState
    {
        DeviceNotAllocated,
        DeviceIsNewer,
        HostIsNewer,
        Synched
    }

    public class GpuMatrix
    {
        public GpuMatrix(Matrix<float> matrix,
                         IntPtr devPtr, DevHostState devHostState)
        {
            Matrix = matrix;
            DevHostState = devHostState;
            DevPtr = devPtr;
        }

        public Matrix<float> Matrix { get; }
        public DevHostState DevHostState { get; }
        public IntPtr DevPtr { get; }
    }

    public static class GpuMatrixOps
    {
        public static string SetupGpuMatrix(out GpuMatrix gmOut, Matrix<float> mtxIn)
        {
            var gpuIn = new GpuMatrix(
                    matrix: mtxIn,
                    devPtr: new IntPtr(),
                    devHostState: DevHostState.DeviceNotAllocated);

            GpuMatrix gpuDeviced;
            var res = GpuMatrixOps.AllocateOnDevice(out gpuDeviced, gpuIn);
            res = res + GpuMatrixOps.CopyToDevice(out gmOut, gpuDeviced);
            return res;
        }

        public static string AllocateOnDevice(out GpuMatrix gmOut, GpuMatrix gmIn)
        {
            IntPtr devData = new IntPtr();

            var aa = new CudaArray();
            var strRet = aa.MallocFloatsOnDevice(ref devData, (uint)gmIn.Matrix.Data.Length);
            if (!String.IsNullOrEmpty(strRet))
            {
                gmOut = null;
                return strRet;
            }
            gmOut = new GpuMatrix(
                            matrix: gmIn.Matrix,
                            devPtr: devData,
                            devHostState: DevHostState.HostIsNewer);

            return String.Empty;

        }

        public static string ClearOnDevice(out GpuMatrix gmOut, GpuMatrix gmIn)
        {
            if (gmIn.DevHostState == DevHostState.DeviceNotAllocated)
            {
                gmOut = null;
                return "Device data pointer already cleared";
            }

            var aa = new CudaArray();
            var strRet = aa.ReleaseDevicePtr(gmIn.DevPtr);
            if (!String.IsNullOrEmpty(strRet))
            {
                gmOut = null;
                return strRet;
            }

            gmOut = new GpuMatrix(
                            matrix: gmIn.Matrix,
                            devPtr: new IntPtr(),
                            devHostState: DevHostState.DeviceNotAllocated);

            return String.Empty;
        }

        public static string CopyToDevice(out GpuMatrix gmOut, GpuMatrix gmIn)
        {
            if (gmIn.DevHostState == DevHostState.DeviceNotAllocated)
            {
                gmOut = null;
                return "Device data pointer not allocated";
            }

            var aa = new CudaArray();
            var strRet = aa.CopyFloatsToDevice(
                                gmIn.Matrix.Data.ToArray(),
                                gmIn.DevPtr, (uint)gmIn.Matrix.Data.Length);

            if (!String.IsNullOrEmpty(strRet))
            {
                gmOut = null;
                return strRet;
            }

            gmOut = new GpuMatrix(
                        matrix: gmIn.Matrix,
                        devPtr: gmIn.DevPtr,
                        devHostState: DevHostState.Synched);

            return String.Empty;
        }

        public static string CopyToHost(out GpuMatrix gmOut, GpuMatrix gmIn)
        {
            if (gmIn.DevHostState == DevHostState.DeviceNotAllocated)
            {
                gmOut = null;
                return "Device data pointer not allocated";
            }

            var hostData = new float[gmIn.Matrix.Data.Length];
            var aa = new CudaArray();

            var strRet = aa.CopyFloatsFromDevice(hostData,
                              gmIn.DevPtr, (uint)gmIn.Matrix.Data.Length);

            if (!String.IsNullOrEmpty(strRet))
            {
                gmOut = null;
                return strRet;
            }

            gmOut = new GpuMatrix(
                matrix: new Matrix<float>(_rows: gmIn.Matrix.Rows,
                                        _cols: gmIn.Matrix.Cols,
                                        host_data: ImmutableArray.Create(hostData),
                                        matrixFormat: MatrixFormat.Column_Major),
                devPtr: gmIn.DevPtr,
                devHostState: DevHostState.Synched);

            return String.Empty;
        }

        //C = α op ( A ) op ( B ) + β C
        public static string Multiply(
            out GpuMatrix gmOut, IntPtr cublasHandle,
            GpuMatrix gmA, GpuMatrix gmB, GpuMatrix gmC)
        {
            if (gmA.DevHostState == DevHostState.DeviceNotAllocated)
            {
                gmOut = null;
                return "Device data pointer for matrix A not allocated";
            }
            if (gmB.DevHostState == DevHostState.DeviceNotAllocated)
            {
                gmOut = null;
                return "Device data pointer for matrix B not allocated";
            }
            if (gmC.DevHostState == DevHostState.DeviceNotAllocated)
            {
                gmOut = null;
                return "Device data pointer for matrix C not allocated";
            }

            var cuby = new CublasClr.Cublas();
            var strRet = cuby.cublasSgemm(
                    cublas_handle: cublasHandle,
                    transa: CublasOp.N,
                    transb: CublasOp.N,
                    m: (int)gmA.Matrix.Rows,
                    n: (int)gmB.Matrix.Cols,
                    k: (int)gmA.Matrix.Cols,
                    alpha: 1,
                    dev_A: gmA.DevPtr,
                    lda: (int)gmA.Matrix.Rows,
                    dev_B: gmB.DevPtr,
                    ldb: (int)gmB.Matrix.Rows,
                    beta: 0,
                    dev_C: gmC.DevPtr,
                    ldc: (int)gmC.Matrix.Rows
                );


            if (!String.IsNullOrEmpty(strRet))
            {
                gmOut = null;
                return strRet;
            }

            gmOut = new GpuMatrix(
                matrix: gmC.Matrix,
                devPtr: gmC.DevPtr,
                devHostState: DevHostState.DeviceIsNewer);

            return String.Empty;
        }

    }

}
