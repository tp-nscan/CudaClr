using Sponge.ViewModel.Common;
using Sponge.Model;
using Utils;
using System.Drawing;
using System.Drawing.Imaging;
using Accord.Video.FFMPEG;
using Sponge.ViewModel.Ga;
using Sponge.ViewModel.Lattice;

///     ///     ///     ///     ///     ///     ///     ///     ///     ///     ///     ///     ///     ///     /// 
/// Install-Package Accord.Video.FFMPEG -Version 3.8.0
///     ///     ///     ///     ///     ///     ///     ///     ///     ///     ///     ///     ///     /// 
///     
namespace Sponge
{
    public partial class MainWindow
    {
        private const int PicSpan = 1000;
        private readonly UpdateGRVm _updateGGVm;
        private readonly UpdateGridVm _updateGridVm;
        private readonly UpdateGGRVm _updateGGRVm;
        private readonly UpdateGaVm _updateGaVm;

        public MainWindow()
        {
            InitializeComponent();
            //UpdateGRControl.DataContext = _updateGGVm = GR_ThermoIsing_bp.Thermo();
            //UpdateGGControl.DataContext = _updateGGVm = GG_Thermo.Thermo();
            //UpdateGridControl.DataContext = _updateGridVm = UpdateGridVmB.BlockPicker();
            //UpdateGGControl.DataContext = _updateGGVm = GG_Annealer.Annealer();
            //UpdateGGControl.DataContext = _updateGGVm = GG_AnnealerRb.AnnealerRb();

            //int width = 320;
            //int height = 240;
            //var writer = new Accord.Video.FFMPEG.VideoFileWriter();

            //writer.Open("test.avi", width, height, 25, VideoCodec.MPEG4);
            //// create a bitmap to save into the video file
            //Bitmap image = new Bitmap(width, height, PixelFormat.Format24bppRgb);
            //// write 1000 video frames
            //for (int i = 0; i < 1000; i++)
            //{
            //    image.SetPixel(i % width, i % height, Color.Red);
            //    writer.WriteVideoFrame(image);
            //}
            //writer.Close();

            // UpdateGGGControl.DataContext = _updateGGRVm = GGR_ThermoIsing_bp.Thermo();

            UpdateGaControl.DataContext = _updateGaVm = UpdateGaVmExt.GaConjOrbitVm();
        }

    }

}
