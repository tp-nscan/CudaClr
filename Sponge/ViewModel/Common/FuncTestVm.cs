using System;
using System.Diagnostics;
using System.Threading;
using System.Threading.Tasks;
using System.Windows.Input;
using Sponge.Common;
using CuArrayClr;
using GridProcsClr;
using System.Linq;
using Utils;

namespace Sponge.ViewModel.Common
{

    public class FuncTestVm : BindableBase
    {
        private bool _isRunning;
        public bool IsRunning
        {
            get { return _isRunning; }
            set
            {
                SetProperty(ref _isRunning, value);
            }
        }

        private bool CanStart()
        {
            return !_isRunning;
        }


        #region local vars
        private CancellationTokenSource _cancellationTokenSource = new CancellationTokenSource();
        private readonly Stopwatch _stopwatch = new Stopwatch();

        #endregion

        #region StepCommand

        private RelayCommand _stepCommand;

        public ICommand StepCommand => _stepCommand ?? (_stepCommand = new RelayCommand(
            Funky,
            CanStart
            ));


        private async void DoStep()
        {
            _cancellationTokenSource = new CancellationTokenSource();
            _isRunning = true;
            CommandManager.InvalidateRequerySuggested();

            await Task.Run(() =>
            {
                _stopwatch.Reset();
                _stopwatch.Start();
              ///  GridValues = Proc();
            },
            _cancellationTokenSource.Token);

            _isRunning = false;
            CommandManager.InvalidateRequerySuggested();
           // UpdateUI(_stopwatch.ElapsedMilliseconds / 1000.0, GpuSteps);
        }


        private void Funky()
        {
            uint span = 32;
            uint arrayLen = span* span;

            var aP0 = ArrayGen.RandInts3(seed: 123, span: span, blockSize: span/2, fracOnes: 0.4);
            var aP1 = Enumerable.Repeat<uint>(0, (int)arrayLen).ToArray();
            var aa = new CudaArray();
            var gp = new GridProcs();
            IntPtr caPlane0 = new IntPtr();
            IntPtr caPlane1 = new IntPtr();

            var retlist = new int[(int)arrayLen];

            try
            {
                var res = aa.ResetDevice();
                res = res + aa.MallocIntsOnDevice(ref caPlane0, arrayLen);
                res = res + aa.CopyIntsToDevice(aP0, caPlane0, arrayLen);
                res = res + aa.CopyIntsFromDevice(retlist, caPlane0, arrayLen);

                if (!aP0.SequenceEqual(retlist))
                {
                   var s = "fail: sequences do not match";
                }

                if (res != String.Empty)
                {
                    //  return testName + " fail: " + res;
                }
                // return testName + " pass";
            }
            catch (Exception ex)
            {
                // return testName + " exception " + ex.Message;
            }
            finally
            {
                aa.ReleaseDevicePtr(caPlane0);
                aa.ResetDevice();
            }
        }

        private void Funky2()
        {
            string testName = "TestCopyIntsToDevice";
            uint arrayLen = 10000;
            var alist = Enumerable.Range(4, (int)arrayLen).ToArray();
            var aa = new CudaArray();
            var gp = new GridProcs();
            System.IntPtr devData = new IntPtr();
            var retlist = new int[(int)arrayLen];

            try
            {
                var res = aa.ResetDevice();
                res = res + aa.MallocIntsOnDevice(ref devData, arrayLen);
                res = res + aa.CopyIntsToDevice(alist, devData, arrayLen);
                res = res + aa.CopyIntsFromDevice(retlist, devData, arrayLen);

                if (!alist.SequenceEqual(retlist))
                {
                   // return testName + " fail: sequences do not match";
                }

                if (res != String.Empty)
                {
                  //  return testName + " fail: " + res;
                }
               // return testName + " pass";
            }
            catch (Exception ex)
            {
               // return testName + " exception " + ex.Message;
            }
            finally
            {
                aa.ReleaseDevicePtr(devData);
                aa.ResetDevice();
            }
        }

        #endregion

    }
}
