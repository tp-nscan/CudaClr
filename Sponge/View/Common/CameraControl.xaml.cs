using System;
using System.IO;
using System.Globalization;
using System.Windows;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using Utils;

namespace Sponge.View.Common
{
    public partial class CameraControl
    {
        public CameraControl()
        {
            InitializeComponent();
        }

        #region TargetVisual

        public static readonly DependencyProperty TargetVisualProperty = DependencyProperty.Register(
                "TargetVisual", typeof(Visual), typeof(CameraControl),
                new FrameworkPropertyMetadata(null, FrameworkPropertyMetadataOptions.AffectsRender));

        public static void SetTargetVisual(DependencyObject depObj, Visual visual)
        {
            depObj.SetValue(TargetVisualProperty, visual);
        }

        public static Visual DefaultVisual()
        {
            var text = new FormattedText(
                textToFormat: "The camera's TargetVisualProperty was not set",
                culture: new CultureInfo("en-us"),
                flowDirection: FlowDirection.LeftToRight,
                typeface: new Typeface(new FontFamily(), FontStyles.Normal, FontWeights.Normal, new FontStretch()),
                emSize: 12,
                foreground: Brushes.Red);

            var drawingVisual = new DrawingVisual();
            var drawingContext = drawingVisual.RenderOpen();
            drawingContext.DrawText(text, new Point(2, 2));
            drawingContext.Close();

            return drawingVisual;
        }

        public static Visual GetTargetVisual(DependencyObject depObj)
        {
            return (Visual)depObj.GetValue(TargetVisualProperty);
        }

        #endregion //TargetVisual

        #region Bitmap

        public static readonly DependencyProperty BitmapEncoderProperty = DependencyProperty.Register(
                    "BitmapEncoder", typeof(PngBitmapEncoder), typeof(CameraControl),
                    new FrameworkPropertyMetadata(null, FrameworkPropertyMetadataOptions.AffectsRender));

        public static void SetBitmapEncoder(DependencyObject depObj, PngBitmapEncoder visual)
        {
            depObj.SetValue(BitmapEncoderProperty, visual);
        }

        public static PngBitmapEncoder GetBitmapEncoder(DependencyObject depObj)
        {
            return (PngBitmapEncoder)depObj.GetValue(BitmapEncoderProperty);
        }

        public PngBitmapEncoder BitmapEncoder
        {
            get { return GetBitmapEncoder(this); }
            set { SetBitmapEncoder(this, value); }
        }

        #endregion //BitmapEncoder

        private void ButtonBase_OnClick(object sender, RoutedEventArgs e)
        {
            var visual = (FrameworkElement)GetTargetVisual(this);
            var bmp = new RenderTargetBitmap((int)visual.ActualWidth, (int)visual.ActualHeight, 96, 96, PixelFormats.Pbgra32);
            bmp.Render(visual);

            var encoder = new PngBitmapEncoder();
            var frame = BitmapFrame.Create(bmp);
            encoder.Frames.Add(frame);

            // BitmapEncoder = encoder;

            var filePath =
                AppDomain.CurrentDomain.BaseDirectory +
                "ScreenShots" +
                Path.DirectorySeparatorChar +
                DateTime.Now.ToDateFileFormat();


            Directory.CreateDirectory(filePath);
            var fileName =
                filePath +
                Path.DirectorySeparatorChar +
                "Pic_" + DateTime.Now.ToDateTimeFileFormat();

            using (var stream = File.Create(fileName + ".png"))
            {
                encoder.Save(stream);
            }
        }
    }
}
