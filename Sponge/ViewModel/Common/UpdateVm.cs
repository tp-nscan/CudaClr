using System;
using System.Reactive.Subjects;
using System.Threading;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Input;
using System.Windows.Threading;
using Sponge.Common;
using Sponge.Model;

namespace Sponge.ViewModel.Common
{
    public class UpdateVm : BindableBase
    {
        private readonly Subject<ProcResult> _updateUI 
            = new Subject<ProcResult>();
        public IObservable<ProcResult> OnUpdateUI => _updateUI;

        public UpdateVm(Func<int, ProcResult> proc)
        {
            Proc = proc;
        }

        public Func<int, ProcResult> Proc { get; private set; }

        string _errorMsg;
        public string ErrorMsg
        {
            get { return _errorMsg; }
            protected set { _errorMsg = value; }
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

        private int _totalSteps;
        public int TotalSteps
        {
            get { return _totalSteps; }
            protected set
            {
                SetProperty(ref _totalSteps, value);
            }
        }

        private int _stepsPerUpdate;
        public int StepsPerUpdate
        {
            get { return _stepsPerUpdate; }
            set
            {
                SetProperty(ref _stepsPerUpdate, value);
            }
        }

        private double _time;
        public double Time
        {
            get { return _time; }
            protected set
            {
                SetProperty(ref _time, value);
            }
        }

        #region local vars

        private CancellationTokenSource _cancellationTokenSource 
            = new CancellationTokenSource();

        #endregion
  

        #region StepCommand

        private RelayCommand _stepCommand;

        public ICommand StepCommand => _stepCommand ?? (
            _stepCommand = new RelayCommand(DoStep, CanStart));


        private async void DoStep()
        {
            _cancellationTokenSource = new CancellationTokenSource();
            IsRunning = true;
            CommandManager.InvalidateRequerySuggested();

            await Task.Run(() =>
            {
                var res = Proc(StepsPerUpdate);
                Application.Current.Dispatcher.Invoke
                    (
                        () => UpdateUI(res),
                        DispatcherPriority.Background
                    );
            },
            _cancellationTokenSource.Token);

            IsRunning = false;
            CommandManager.InvalidateRequerySuggested();
        }

        #endregion

        #region StartCommand

        private RelayCommand _startCommand;

        public ICommand StartCommand => _startCommand ?? (
            _startCommand = new RelayCommand(DoStart, CanStart));

        private async void DoStart()
        {
            _cancellationTokenSource = new CancellationTokenSource();
            IsRunning = true;
            CommandManager.InvalidateRequerySuggested();

            string errorMsg = string.Empty;

            await Task.Run(() =>
            {
                for (var i = 0; (IsRunning & string.IsNullOrEmpty(errorMsg)); i++)
                {
                    var res = Proc(StepsPerUpdate);
                    errorMsg = res.ErrorMsg;
                    Application.Current.Dispatcher.Invoke
                        (
                            () => UpdateUI(res),
                            DispatcherPriority.Background
                        );

                    if (_cancellationTokenSource.IsCancellationRequested)
                    {
                        IsRunning = false;
                        CommandManager.InvalidateRequerySuggested();
                    }
                }

            },
            _cancellationTokenSource.Token);
        }

        void UpdateUI(ProcResult result)
        {
            TotalSteps += result.StepsCompleted;
            Time += (result.TimeInMs / 1000);
            ErrorMsg = result.ErrorMsg;
            _updateUI.OnNext(result);
        }

        private bool CanStart()
        {
            return !IsRunning;
        }

        #endregion // StartCommand

        #region StopCommand

        private RelayCommand _stopCommand;

        public ICommand StopCommand => _stopCommand ?? (
            _stopCommand = new RelayCommand(DoStop, CanStop));

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

        bool _resetIssued;
        bool ResetIssued
        {
            set { _resetIssued = value; }
            get { return _resetIssued; }
        }

        private RelayCommand _resetCommand;

        public ICommand ResetCommand => _resetCommand ?? (
            _resetCommand = new RelayCommand(DoReset, CanReset));


        private void DoReset()
        {
            ResetIssued = true;
            _cancellationTokenSource.Cancel();
        }

        private bool CanReset()
        {
            return true; // _isRunning;
        }

        #endregion

    }
}
