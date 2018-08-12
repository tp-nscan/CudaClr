using Sponge.ViewModel.Common;
using Sponge.Model;

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
            //UpdateGridControl.DataContext = _updateGridVm = UpdateGridVmB.BlockPicker();
            UpdateGGControl.DataContext = _updateGGVm = GG_Thermo.Thermo(); // GG_Annealer.Annealer();
        }

    }

}
