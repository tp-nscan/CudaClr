
using System.Windows;
using FS;

namespace Sponge.View.Common
{
    public partial class LegendControl
    {
        public LegendControl()
        {
            InitializeComponent();
        }

        public double ImageHeight
        {
            get { return (double)GetValue(ImageHeightProperty); }
            set
            {
                SetValue(ImageHeightProperty, value);
            }
        }

        public static readonly DependencyProperty ImageHeightProperty =
            DependencyProperty.Register("ImageHeight", typeof(double), typeof(LegendControl),
                new PropertyMetadata(100.0));
    }
}
