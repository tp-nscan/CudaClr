using Sponge.ViewModel.Common;

namespace Sponge
{
    public partial class MainWindow
    {
        private const int PicSpan = 1000;

        private readonly FuncTestVm _funcTestVm;
        //private readonly UpdateBitsVm _updateGridVm;
        //private readonly UpdateClockVm _updateGridVm;

        public MainWindow()
        {
            InitializeComponent();
            FuncTestControl.DataContext = _funcTestVm = new FuncTestVm();
            //_updateGridVm = new UpdateClockVm();
            //UpdateClockControl.DataContext = _updateGridVm;

            //_updateGridVm = new UpdateBitsVm();
            //UpdateBitsControl.DataContext = _updateGridVm;

        }

    }
}
