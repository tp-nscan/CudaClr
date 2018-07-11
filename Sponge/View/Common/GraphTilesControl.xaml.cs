using Sponge.ViewModel.Common;

namespace Sponge.View.Common
{
    public partial class GraphTilesControl
    {
        public GraphTilesControl()
        {
            InitializeComponent();
            this.DataContextChanged += GraphTilesControl_DataContextChanged;
        }

        private void GraphTilesControl_DataContextChanged(object sender, System.Windows.DependencyPropertyChangedEventArgs e)
        {
            if (DataContext is GraphTilesVm)
            {
                ((GraphTilesVm)DataContext).ImageSize = new FS.Sz2<double>(TilesControl.ActualWidth, TilesControl.ActualHeight);
            }
        }

        private void TilesControl_SizeChanged(object sender, System.Windows.SizeChangedEventArgs e)
        {
            if (DataContext is GraphTilesVm)
            {
                ((GraphTilesVm)DataContext).ImageSize = new FS.Sz2<double>(e.NewSize.Width, e.NewSize.Height);
            }
        }
    }
}
