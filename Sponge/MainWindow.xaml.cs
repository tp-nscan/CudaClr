using Sponge.ViewModel.Common;
using Sponge.M2;

namespace Sponge
{
    public partial class MainWindow
    {
        private const int PicSpan = 1000;

        //private readonly FuncTestVm _funcTestVm;
       // private readonly UpdateBitsVm _updateBitsVm;
        //private readonly UpdateClockVm _updateClockVm;
        //private readonly UpdateContVm _updateContVm;
        private readonly UpdateBinVm _updateBinVm;

        public MainWindow()
        {
            InitializeComponent();
            //FuncTestControl.DataContext = _funcTestVm = new FuncTestVm();
            //_updateClockVm = new UpdateClockVm();
            //UpdateClockControl.DataContext = _updateGridVm;

            //UpdateBitsControl.DataContext = _updateBitsVm= new UpdateBitsVm();
            //_updateContVm = new UpdateContVm();
            //UpdateClockControl.DataContext = _updateContVm = new UpdateContVm();
            ProcBinControl.DataContext = _updateBinVm = new UpdateBinVm(SimGridSamples.TestInt2());
        }

    }
}
