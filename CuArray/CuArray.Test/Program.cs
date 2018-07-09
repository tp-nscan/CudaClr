using System;
using System.Linq;
using CuArrayClr;


namespace CuArray.Test
{
    class Program
    {
        static void Main(string[] args)
        {
            var aa = new CudaArray();
            var res = aa.TestRuntimeErr();
            res = aa.TestCudaStatusErr();

            Console.WriteLine(TestCopyIntsToDevice());
            Console.WriteLine(TestCopyFloatsToDevice());
            Console.WriteLine(TestCopyIntsDeviceToDevice());
            Console.WriteLine(TestCopyFloatsDeviceToDevice());
        }

        static string TestCopyIntsToDevice()
        {
            string testName = "TestCopyIntsToDevice";
            uint arrayLen = 1000;
            var alist = Enumerable.Range(4, (int)arrayLen).ToArray();
            var aa = new CudaArray();
            System.IntPtr devData = new System.IntPtr();
            var retlist = new int[(int)arrayLen];
             
            try
            {
                var res = aa.ResetDevice();
                res = res + aa.MallocIntsOnDevice(ref devData, arrayLen);
                res = res + aa.CopyIntsToDevice(alist, devData, arrayLen);
                res = res + aa.CopyIntsFromDevice(retlist, devData, arrayLen);

                if(! alist.SequenceEqual(retlist))
                {
                    return testName + " fail: sequences do not match";
                }

                if (res != String.Empty)
                {
                    return testName + " fail: " + res;
                }
                return testName + " pass";
            }
            catch (Exception ex)
            {
                return testName + " exception " + ex.Message;
            }
            finally
            {
                aa.ReleaseDevicePtr(devData);
                aa.ResetDevice();
            }
        }

        static string TestCopyFloatsToDevice()
        {
            string testName = "TestCopyFloatsToDevice";
            uint arrayLen = 1000;
            var alist = Enumerable.Range(4, (int)arrayLen).Select(t => (float)t).ToArray();
            var aa = new CudaArray();
            IntPtr devData = new System.IntPtr();
            var retlist = new float[(int)arrayLen];

            try
            {
                var res = aa.ResetDevice();
                res = res + aa.MallocIntsOnDevice(ref devData, arrayLen);
                res = res + aa.CopyFloatsToDevice(alist, devData, arrayLen);
                res = res + aa.CopyFloatsFromDevice(retlist, devData, arrayLen);

                if (!alist.SequenceEqual(retlist))
                {
                    return testName + " fail: sequences do not match";
                }

                if (res != String.Empty)
                {
                    return testName + " fail: " + res;
                }
                return testName + " pass";
            }
            catch (Exception ex)
            {
                return testName + " exception " + ex.Message;
            }
            finally
            {
                aa.ReleaseDevicePtr(devData);
                aa.ResetDevice();
            }
        }

        static string TestCopyIntsDeviceToDevice()
        {
            string testName = "TestCopyIntsDeviceToDevice";
            uint arrayLen = 1000;
            var alist = Enumerable.Range(4, (int)arrayLen).ToArray();
            var aa = new CudaArray();
            IntPtr devDataA = new System.IntPtr();
            IntPtr devDataB = new System.IntPtr();
            var retlist = new int[(int)arrayLen];

            try
            {
                var res = aa.ResetDevice();
                res = res + aa.MallocIntsOnDevice(ref devDataA, arrayLen);
                res = res + aa.MallocIntsOnDevice(ref devDataB, arrayLen);
                res = res + aa.CopyIntsToDevice(alist, devDataA, arrayLen);
                res = res + aa.CopyIntsDeviceToDevice(devDataB, devDataA, arrayLen);
                res = res + aa.CopyIntsFromDevice(retlist, devDataB, arrayLen);
                res = res + aa.ReleaseDevicePtr(devDataA);
                res = res + aa.ReleaseDevicePtr(devDataB);

                if (!alist.SequenceEqual(retlist))
                {
                    return testName + " fail: sequences do not match";
                }

                if (res != String.Empty)
                {
                    return testName + " fail: " + res;
                }
                return testName + " pass";
            }
            catch (Exception ex)
            {
                return testName + " exception " + ex.Message;
            }
            finally
            {
                aa.ReleaseDevicePtr(devDataA);
                aa.ReleaseDevicePtr(devDataB);
                aa.ResetDevice();
            }
        }

        static string TestCopyFloatsDeviceToDevice()
        {
            string testName = "TestCopyFloatsDeviceToDevice";
            uint arrayLen = 1000;
            var alist = Enumerable.Range(4, (int)arrayLen).Select(t => (float)t).ToArray();
            var aa = new CudaArray();
            IntPtr devDataA = new System.IntPtr();
            IntPtr devDataB = new System.IntPtr();
            var retlist = new float[(int)arrayLen];

            try
            {
                var res = aa.ResetDevice();
                res = res + aa.MallocFloatsOnDevice(ref devDataA, arrayLen);
                res = res + aa.MallocFloatsOnDevice(ref devDataB, arrayLen);
                res = res + aa.CopyFloatsToDevice(alist, devDataA, arrayLen);
                res = res + aa.CopyFloatsDeviceToDevice(devDataB, devDataA, arrayLen);
                res = res + aa.CopyFloatsFromDevice(retlist, devDataB, arrayLen);
                res = res + aa.ReleaseDevicePtr(devDataA);
                res = res + aa.ReleaseDevicePtr(devDataB);

                if (!alist.SequenceEqual(retlist))
                {
                    return testName + " fail: sequences do not match";
                }

                if (res != String.Empty)
                {
                    return testName + " fail: " + res;
                }
                return testName + " pass";
            }
            catch(Exception ex)
            {
                return testName + " exception " + ex.Message;
            }
            finally
            {
                aa.ReleaseDevicePtr(devDataA);
                aa.ReleaseDevicePtr(devDataB);
                aa.ResetDevice();
            }
        }

    }
}
