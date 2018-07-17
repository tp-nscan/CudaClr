using System;
using System.Linq;
using CuArrayClr;

namespace Rando.Test
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine(TestMakeUniformRands());
            Console.WriteLine(TestMakeNormalRands());
           // Console.WriteLine(TestMakeRandomInts());
        }

        static string TestMakeRandomInts()
        {
            string testName = "TestMakeRandomInts";
            uint arrayLen = 1000;
            int seed = 1234;
            var aa = new CudaArray();
            var rdo = new RandoClr.RandoProcs();
            IntPtr devRando = new IntPtr();
            IntPtr devData = new IntPtr();
            var retlist = new int[(int)arrayLen];

            try
            {
                var res = aa.ResetDevice();

                res = res + rdo.MakeGenerator32(ref devRando, seed);
                res = res + aa.MallocIntsOnDevice(ref devData, arrayLen);
                res = res + rdo.MakeRandomInts(devData, devRando, arrayLen);
                res = res + aa.CopyIntsFromDevice(retlist, devData, arrayLen);

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

        static string TestMakeUniformRands()
        {
            string testName = "TestMakeUniformRands";
            uint arrayLen = 100000;
            int seed = 1234;
            var aa = new CudaArray();
            var rdo = new RandoClr.RandoProcs();
            IntPtr devRando = new IntPtr();
            IntPtr devData = new IntPtr();
            var retlist = new float[arrayLen];

            try
            {
                var res = aa.ResetDevice();
 
                res = res + rdo.MakeGenerator64(ref devRando, seed);
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
            var rdo = new RandoClr.RandoProcs();
            var aa = new CudaArray();
            uint arrayLen = 1000;
            int seed = 1234;
            IntPtr devRando = new IntPtr();
            IntPtr devData = new IntPtr();
            var retlist = new float[(int)arrayLen];

            try
            {
                var res = aa.ResetDevice();

                res = res + rdo.MakeGenerator64(ref devRando, seed);
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
