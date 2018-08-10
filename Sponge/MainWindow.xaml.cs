using Sponge.ViewModel.Common;
using Sponge.Model;

namespace Sponge
{
    public partial class MainWindow
    {
        private const int PicSpan = 1000;
        // private readonly UpdateIsingBpVm _updateIsingBpVm;
        // private readonly UpdateGridVm _updateGridVm;
         private readonly UpdateIsing2dVm _updateIsing2dVm;
        // private readonly UpdateDualIsingVm _updateDualIsingVm;
        // private readonly UpdateIsingDualTempVm _updateIsingDualTempVm;

        public MainWindow()
        {
            InitializeComponent();


           // UpdateIsingControl.DataContext = _updateIsingBpVm = new UpdateIsingBpVm(SimGridSamples.SquareRandBits(1024, 123));
            UpdateIsingControl.DataContext = _updateIsing2dVm = new UpdateIsing2dVm(SimGridSamples.SquareRandBits(512, 5213));
            //UpdateDualIsingControl.DataContext = _updateDualIsingVm = new UpdateDualIsingVm(SimGridSamples.SquareRingBits(512));
            //UpdateIsingDualTempControl.DataContext = _updateIsingDualTempVm = new UpdateIsingDualTempVm(SimGridSamples.SquareRingBits(1024));
        }

    }
}
