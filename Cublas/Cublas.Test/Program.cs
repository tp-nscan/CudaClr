using System;
using CuArrayClr;
using CublasClr;

namespace Cublas.Test
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine(TestcublasSgemm2());
            //Console.WriteLine(TestCublasHandle());
            //Console.WriteLine(TestcublasSgemm());
            //Console.WriteLine(TestMakeNormalRands());
        }

        static string TestRowMajToColMaj()
        {

            var dataA = GpuMatrixUtils.AA();
            var transf = GpuMatrixUtils.RowMajorToColMajor(dataA, 3, 2);

            return String.Empty;
        }

        static string TestGpuMatrix()
        {
            var data = CublasClr.GpuMatrixUtils.MakeColumnMajor(rows: 5, cols: 3);
            GpuMatrix gpuSetup;
            var res = GpuMatrixOps.SetupGpuMatrix(out gpuSetup, data, rows: 5, cols: 3);
            GpuMatrix gpuReturned;
            res = res + GpuMatrixOps.CopyToHost(out gpuReturned, gpuSetup);
            GpuMatrix gpuCleared;
            res = res + GpuMatrixOps.ClearOnDevice(out gpuCleared, gpuReturned);

            var dout = gpuReturned.Matrix.Data;
            return String.Empty;
        }

        static string TestCublasHandle()
        {
            CublasOp o = CublasClr.CublasOp.N;

            string testName = "TestCublasHandle";
            var cuby = new CublasClr.Cublas();
            IntPtr devHandle = new IntPtr();
            var aa = new CudaArray();
            try
            {
                var res = aa.ResetDevice();
                res = res + cuby.MakeCublasHandle(ref devHandle);
                res = res + cuby.DestroyCublasHandle(devHandle);
                if (res != String.Empty)
                {
                    return testName + " fail: " + res;
                }
                return testName + " pass";
            }
            catch
            {
                return testName + " fail";
            }
            finally
            {
                //aa.ReleaseDevicePtr(devData);
                aa.ResetDevice();
            }
        }

        static string TestcublasSgemm1()
        {
            string testName = "TestcublasSgemm";
            uint aw = 5;
            uint bh = aw;
            uint ah = 5;
            uint bw = 5;
            uint ch = ah;
            uint cw = bw;

            var cuby = new CublasClr.Cublas();
            var aa = new CudaArray();
            var res = aa.ResetDevice();

            var dataA = GpuMatrixUtils.MakeIdentity(rows: ah, cols: aw);
            GpuMatrix gpuA;
            res = res + GpuMatrixOps.SetupGpuMatrix(out gpuA, dataA, rows: ah, cols: aw);

            var dataB = GpuMatrixUtils.MakeIdentiPoke(rows: bh, cols: bw);
            GpuMatrix gpuB;
            res = res + GpuMatrixOps.SetupGpuMatrix(out gpuB, dataB, rows: bh, cols: bw);

            var dataC = GpuMatrixUtils.MakeZeroes(rows: bh, cols: bw);
            GpuMatrix gpuC;
            res = res + GpuMatrixOps.SetupGpuMatrix(out gpuC, dataC, rows: ch, cols: cw);

            IntPtr cublasHandle = new IntPtr();
            res = res + cuby.MakeCublasHandle(ref cublasHandle);

            GpuMatrix gpuProd;
            res = res + GpuMatrixOps.Multiply(
                gmOut: out gpuProd,
                cublasHandle: cublasHandle,
                gmA: gpuA, gmB: gpuB, gmC: gpuC);

            GpuMatrix gpuSynched;
            res = res + GpuMatrixOps.CopyToHost(out gpuSynched, gpuProd);

            var cpuRes = new float[ah * bw];
            GpuMatrixUtils.MatrixMult(C: cpuRes, A: dataA, B: dataB, wA: aw, hA: ah, wB: bw);

            var cpuRes2 = new float[bh * aw];
            GpuMatrixUtils.MatrixMult(C: cpuRes2, A: dataB, B: dataA, wA: bw, hA: bh, wB: aw);

            return res;
        }

        static string TestcublasSgemm2()
        {
            string testName = "TestcublasSgemm2";
            uint aw = 2;
            uint bh = aw;
            uint ah = 3;
            uint bw = 3;
            uint ch = ah;
            uint cw = bw;
            GpuMatrix gpuA;
            GpuMatrix gpuB;
            GpuMatrix gpuC;

            var dataA = GpuMatrixUtils.AA();
            var dataB = GpuMatrixUtils.BB();
            var cRes = new float[ch * cw];

            var cuby = new CublasClr.Cublas();
            var aa = new CudaArray();

            var res = aa.ResetDevice();
            res = res + GpuMatrixOps.SetupGpuMatrix(out gpuA, dataA, rows: ah, cols: aw);
            res = res + GpuMatrixOps.SetupGpuMatrix(out gpuB, dataB, rows: bh, cols: bw);
            res = res + GpuMatrixOps.SetupGpuMatrix(out gpuC, cRes, rows: ch, cols: cw);


            IntPtr cublasHandle = new IntPtr();
            res = res + cuby.MakeCublasHandle(ref cublasHandle);

            GpuMatrix gpuProd;
            res = res + GpuMatrixOps.Multiply(
                gmOut: out gpuProd,
                cublasHandle: cublasHandle,
                gmA: gpuA, gmB: gpuB, gmC: gpuC);

            GpuMatrix gpuSynched;
            res = res + GpuMatrixOps.CopyToHost(out gpuSynched, gpuProd);

            // GpuMatrixUtils.MatrixMult(C: cRes, A: dataA, B: dataB, wA: aw, hA: ah, wB: bw);

            return string.Empty;

        }


        static string TestMakeNormalRands()
        {
            string testName = "TestCopyIntsToDevice";
            //var rdo = new RandoClr.Rando();
            //var aa = new CudaArray();
            //uint arrayLen = 1000;
            //IntPtr devData = new IntPtr();
            //var alist = Enumerable.Range(4, (int)arrayLen).ToArray();
            //var retlist = Enumerable.Repeat(0, (int)arrayLen).ToArray();

            try
            {
                //var res = aa.ResetDevice();
                //res = res + aa.MallocIntsOnDevice(ref devData, arrayLen);
                //res = res + aa.CopyIntsToDevice(alist, devData, arrayLen);
                //res = res + aa.CopyIntsFromDevice(retlist, devData, arrayLen);

                //res = res + aa.ReleaseDevicePtr(devData);
                //if (res != String.Empty)
                //{
                //    return testName + " fail: " + res;
                //}
                return testName + " pass";
            }
            catch
            {
                return testName + " fail";
            }
            finally
            {
                //aa.ReleaseDevicePtr(devData);
                //aa.ResetDevice();
            }
        }
    }
}
