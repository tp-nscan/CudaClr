using Sponge.ViewModel.Common;

namespace Sponge
{
    public partial class MainWindow
    {
        private const int PicSpan = 1000;

        //private readonly FuncTestVm _funcTestVm;
        //private readonly UpdateBitsVm _updateBitsVm;
        //private readonly UpdateClockVm _updateClockVm;
        private readonly UpdateContVm _updateContVm;


        public MainWindow()
        {
            InitializeComponent();
            //FuncTestControl.DataContext = _funcTestVm = new FuncTestVm();
            //_updateClockVm = new UpdateClockVm();
            //UpdateClockControl.DataContext = _updateGridVm;

            //_updateBitsVm = new UpdateBitsVm();
            //UpdateBitsControl.DataContext = _updateBitsVm;


            _updateContVm = new UpdateContVm();
            UpdateClockControl.DataContext = _updateContVm;

        }

    }
}
