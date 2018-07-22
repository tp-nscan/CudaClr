using System;
using System.Diagnostics;
using System.Threading;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Threading;
using Sponge.Common;
using Sponge.Model;
using FS;
using Utils;
using Sponge.M2;

namespace Sponge.ViewModel.Common
{
    public class UpdateContVm : BindableBase
    {
        //public const int Modulus = 16384;
        public const int Modulus = 1025;
        public const int Seed = 195;

        public UpdateContVm(int colorCount = Modulus, uint gridSpan = 512)
        {
            ColorCount = colorCount;
            GridSpan = gridSpan;
            GpuSteps = 100;
            StepSize = 0.2;
            Noise = 1.0;
            _colorSequenceVm = new ColorSequenceVm(ColorCount);
            _legendVm = LegendVmCont.Standard();
            _colorSequenceVm.OnColorsChanged.Subscribe(HandleNewColors);
            _colorSequenceVm.SendColorsChanged();

            InitGridValues();
        }

        void InitGridValues()
        {
            //GridValues = ArrayGen.Dot(span: GridSpan,  modulus: Modulus);
            //GridValues = ArrayGen.Spot(span: GridSpan, spotSz: 245, modulus: Modulus);
            GridValues = FloatArrayGen.SignedUnitUniformRands(span: (int)GridSpan, seed: Seed);
            //GridValues = ArrayGen.RandInts(123, GridSpan, 0, 256);
            //GridValues = ArrayGen.MultiRing(modD: 8, outerD: 128, span: GridSpan, modulus: Modulus);
            //GridValues = ArrayGen.RandInts(123, GridSpan, 0, Modulus);
            //GridValues = ArrayGen.Uniform(GridSpan * GridSpan, Modulus/2);

            GraphLatticeVm = new GraphLatticeVm(
                                new R<uint>(0, GridSpan, 0, GridSpan),
                                "Title D", "TitleX D", "TitleY D");

            _graphLatticeVm.SetUpdater(DrawGridCell, GridValues);
            IsingCt2.Init(inputs: GridValues, span: GridSpan);
        }

        public int ColorCount { get; }

        public uint GridSpan { get; }

        void HandleNewColors(Color[] c)
        {
            _legendVm.MidColors = c;
            _graphLatticeVm?.Update(GridValues);
        }

        private ColorSequenceVm _colorSequenceVm;
        public ColorSequenceVm ColorSequenceVm
        {
            get { return _colorSequenceVm; }
            set
            {
                SetProperty(ref _colorSequenceVm, value);
            }
        }

        private LegendVmCont _legendVm;
        public LegendVmCont LegendVm
        {
            get { return _legendVm; }
            set
            {
                SetProperty(ref _legendVm, value);
            }
        }

        private GraphLatticeVm _graphLatticeVm;
        public GraphLatticeVm GraphLatticeVm
        {
            get { return _graphLatticeVm; }
            set
            {
                SetProperty(ref _graphLatticeVm, value);
            }
        }

        private bool _isRunning;
        public bool IsRunning
        {
            get { return _isRunning; }
            set
            {
                SetProperty(ref _isRunning, value);
            }
        }

        private double _noise;
        public double Noise
        {
            get { return _noise; }
            set
            {
                SetProperty(ref _noise, value);
            }
        }

        private int _step;
        public int Step
        {
            get { return _step; }
            set
            {
                SetProperty(ref _step, value);
            }
        }

        private double _stepSize;
        public double StepSize
        {
            get { return _stepSize; }
            set
            {
                SetProperty(ref _stepSize, value);
            }
        }

        private int _gpuSteps;
        public int GpuSteps
        {
            get { return _gpuSteps; }
            set
            {
                SetProperty(ref _gpuSteps, value);
            }
        }

        private double _time;
        public double Time
        {
            get { return _time; }
            set
            {
                SetProperty(ref _time, value);
            }
        }

        public float[] GridValues { get; private set; }

        private object DrawGridCell(P2<int> dataLoc, R<double> imagePatch, object data)
        {
            var offset = dataLoc.X + dataLoc.Y * GridSpan;
            var color = LegendVm.ColorVal(GridValues[offset]);
            return new RV<float, Color>(
                minX: (float)imagePatch.MinX,
                maxX: (float)imagePatch.MaxX,
                minY: (float)imagePatch.MinY,
                maxY: (float)imagePatch.MaxY,
                v: color);
        }


        #region local vars
        private CancellationTokenSource _cancellationTokenSource = new CancellationTokenSource();
        private readonly Stopwatch _stopwatch = new Stopwatch();

        #endregion

        string Proc(float[] gridvals)
        {
            return IsingCt2.Update(
                    results: gridvals,
                    steps: GpuSteps,
                    stepSize: (float)StepSize,
                    noise: (float)Noise);
        }

        #region StepCommand

        private RelayCommand _stepCommand;

        public ICommand StepCommand => _stepCommand ?? (
            _stepCommand = new RelayCommand(DoStep, CanStart));


        private async void DoStep()
        {
            _cancellationTokenSource = new CancellationTokenSource();
            _isRunning = true;
            CommandManager.InvalidateRequerySuggested();

            await Task.Run(() =>
            {
                _stopwatch.Reset();
                _stopwatch.Start();
                var str = Proc(GridValues);
            },
            _cancellationTokenSource.Token);

            _isRunning = false;
            CommandManager.InvalidateRequerySuggested();
            UpdateUI(_stopwatch.ElapsedMilliseconds / 1000.0, GpuSteps);
        }

        #endregion

        #region StartCommand

        private RelayCommand _startCommand;

        public ICommand StartCommand => _startCommand ?? (
            _startCommand = new RelayCommand(DoStart, CanStart));

        private async void DoStart()
        {
            _cancellationTokenSource = new CancellationTokenSource();
            _isRunning = true;
            CommandManager.InvalidateRequerySuggested();

            await Task.Run(() =>
            {
                _stopwatch.Start();

                for (var i = 0; _isRunning; i++)
                {
                    var str = Proc(GridValues);

                    if (_graphLatticeVm != null)
                    {
                        Application.Current.Dispatcher.Invoke
                            (
                                () => UpdateUI(_stopwatch.ElapsedMilliseconds / 1000.0, GpuSteps),
                                DispatcherPriority.Background
                            );
                    }

                    if (_cancellationTokenSource.IsCancellationRequested)
                    {
                        _isRunning = false;
                        _stopwatch.Stop();
                        CommandManager.InvalidateRequerySuggested();
                    }
                }

            },
            _cancellationTokenSource.Token);

        }

        void UpdateUI(double time, int more_steps)
        {
            Step += more_steps;
            _graphLatticeVm.Update(GridValues);
            Time = time;
        }

        private bool CanStart()
        {
            return !_isRunning;
        }

        #endregion // StartCommand

        #region StopCommand

        private RelayCommand _stopCommand;

        public ICommand StopCommand => _stopCommand ?? (_stopCommand = new RelayCommand(
            DoStop, CanStop));

        private void DoStop()
        {
            _cancellationTokenSource.Cancel();
        }

        private bool CanStop()
        {
            return true; // _isRunning;
        }

        #endregion // StopCommand

        #region ResetCommand

        private RelayCommand _resetCommand;

        public ICommand ResetCommand => _resetCommand ?? (
            _resetCommand = new RelayCommand(DoReset, CanReset));


        private void DoReset()
        {
            _cancellationTokenSource.Cancel();
            InitGridValues();
            Step = 0;
            Time = 0;
        }

        private bool CanReset()
        {
            return true; // _isRunning;
        }

        #endregion
    }
}
