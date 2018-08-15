using Sponge.ViewModel.Common;
using Sponge.Model;
using Utils;

namespace Sponge
{
    public partial class MainWindow
    {
        private const int PicSpan = 1000;
        private readonly UpdateGGVm _updateGGVm;
        private readonly UpdateGridVm _updateGridVm;


        public MainWindow()
        {
            InitializeComponent();

            var res = FloatFuncs.Betas(32, 10.0f);

            UpdateGGControl.DataContext = _updateGGVm = GG_ThermoIsing_bp.Thermo();

            //UpdateGGControl.DataContext = _updateGGVm = GG_Thermo.Thermo();
            //UpdateGridControl.DataContext = _updateGridVm = UpdateGridVmB.BlockPicker();
            //UpdateGGControl.DataContext = _updateGGVm = GG_Annealer.Annealer();
            //UpdateGGControl.DataContext = _updateGGVm = GG_AnnealerRb.AnnealerRb();
        }

    }

}
