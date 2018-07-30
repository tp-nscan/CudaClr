using Sponge.ViewModel.Common;
using Sponge.Model;

namespace Sponge
{
    public partial class MainWindow
    {
        private const int PicSpan = 1000;

        //private readonly FuncTestVm _funcTestVm;
        //private readonly UpdateBitsVm _updateBitsVm;
        //private readonly UpdateClockVm _updateClockVm;
        //private readonly UpdateContVm _updateContVm;
        //private readonly UpdateGraphLatticeVm _updateGraphLatticeVm;
        private readonly UpdateIsing2dVm _updateGraphVm;

        public MainWindow()
        {
            InitializeComponent();
            //FuncTestControl.DataContext = _funcTestVm = new FuncTestVm();
            //_updateClockVm = new UpdateClockVm();
            //UpdateClockControl.DataContext = _updateGridVm;

            //UpdateBitsControl.DataContext = _updateBitsVm= new UpdateBitsVm();
            //_updateContVm = new UpdateContVm();
            //UpdateClockControl.DataContext = _updateContVm = new UpdateContVm();
            //UpdateGraphLatticeControl.DataContext = _updateGraphLatticeVm = new UpdateGraphLatticeVm(SimGridSamples.TestInt2());
            UpdateIsingControl.DataContext = _updateGraphVm = new UpdateIsing2dVm(SimGridSamples.TestInt2());

        }

    }
}
