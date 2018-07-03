using System;
using System.Linq;
using CuArrayClr;

namespace Rando.Test
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine(TestCopyIntsToDevice());
            Console.WriteLine(TestMakeUniformRands());
            Console.WriteLine(TestMakeNormalRands());
        }

        static string TestCopyIntsToDevice()
        {
            string testName = "TestCopyIntsToDevice";
            var rdo = new RandoClr.Rando();
            var aa = new CudaArray();
            uint arrayLen = 1000;
            IntPtr devData = new IntPtr();
            var alist = Enumerable.Range(4, (int)arrayLen).ToArray();
            var retlist = Enumerable.Repeat(0, (int)arrayLen).ToArray();

            try
            {
                var res = aa.ResetDevice();
                res = res + aa.MallocIntsOnDevice(ref devData, arrayLen);
                res = res + aa.CopyIntsToDevice(alist, devData, arrayLen);
                res = res + aa.CopyIntsFromDevice(retlist, devData, arrayLen);

                res = res + aa.ReleaseDevicePtr(devData);
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
                aa.ReleaseDevicePtr(devData);
                aa.ResetDevice();
            }
        }

        static string TestMakeUniformRands()
        {
            string testName = "TestMakeUniformRands";
            uint arrayLen = 1000;
            int seed = 1234;
            var aa = new CudaArray();
            var rdo = new RandoClr.Rando();
            IntPtr devRando = new IntPtr();
            System.IntPtr devData = new System.IntPtr();
            var retlist = new float[(int)arrayLen];

            try
            {
                var res = aa.ResetDevice();
 
                res = res + rdo.MakeGenerator(ref devRando, seed);
                res = res + aa.MallocFloatsOnDevice(ref devData, arrayLen);
                res = res + rdo.MakeUniformRands(devData, devRando, arrayLen);
                res = res + aa.CopyFloatsFromDevice(retlist, devData, arrayLen);

                res = res + aa.ReleaseDevicePtr(devData);
                res = res + rdo.DestroyGenerator(devRando);
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
                //rdo.DestroyGenerator(devRando);
                aa.ReleaseDevicePtr(devData);
                aa.ResetDevice();
            }
        }

        static string TestMakeNormalRands()
        {
            string testName = "TestMakeNormalRands";
            var rdo = new RandoClr.Rando();
            var aa = new CudaArray();
            uint arrayLen = 1000;
            int seed = 1234;
            IntPtr devRando = new IntPtr();
            IntPtr devData = new IntPtr();
            var retlist = new float[(int)arrayLen];

            try
            {
                var res = aa.ResetDevice();

                res = res + rdo.MakeGenerator(ref devRando, seed);
                res = res + aa.MallocFloatsOnDevice(ref devData, arrayLen);
                res = res + rdo.MakeNormalRands(devData, devRando, arrayLen, 0.0f, 1.0f);
                res = res + aa.CopyFloatsFromDevice(retlist, devData, arrayLen);

                res = res + aa.ReleaseDevicePtr(devData);
                res = res + rdo.DestroyGenerator(devRando);

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
                //rdo.DestroyGenerator(devRando);
                aa.ReleaseDevicePtr(devData);
                aa.ResetDevice();
            }
        }
    }
}
