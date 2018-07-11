
using Sponge.ViewModel.Common;

namespace Sponge.View.Common
{
    public sealed partial class GraphLatticeControl
    {
        public GraphLatticeControl()
        {
            InitializeComponent();
        }

        private void wbImage_SizeChanged(object sender, System.Windows.SizeChangedEventArgs e)
        {
            if (DataContext is GraphLatticeVm)
            {
                ((GraphLatticeVm)DataContext).ImageSize = new FS.Sz2<double>(e.NewSize.Width, e.NewSize.Height);
            }
        }
    }
}
