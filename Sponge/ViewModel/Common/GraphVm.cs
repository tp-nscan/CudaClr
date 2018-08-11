using System.Collections.Generic;
using System.Linq;
using System.Windows.Media;
using Sponge.Common;
using FS;
using Microsoft.FSharp.Core;

namespace Sponge.ViewModel.Common
{
    public class GraphVm : BindableBase
    {
        public GraphVm(R<float> boundingRect)
        {
            WbImageVm = new WbImageVm();
            BoundingRect = boundingRect;

            WbImageVm.ImageData = Id.MakeImageDataAndClip(
                plotPoints: WbImageVm.ImageData.plotPoints,
                plotLines: WbImageVm.ImageData.plotLines,
                filledRects: WbImageVm.ImageData.filledRects,
                openRects: WbImageVm.ImageData.openRects,
                clipRegion: BoundingRect
                );

            MinStrX = BoundingRect.MinX.ToString();
            MaxStrX = BoundingRect.MaxX.ToString();
            MinStrY = BoundingRect.MinY.ToString();
            MaxStrY = BoundingRect.MaxY.ToString();
        }

        public R<float> BoundingRect { get; set; }

        public WbImageVm WbImageVm { get; }
        

        public void SetData(
            IEnumerable<P2V<float, Color>> plotPoints,
            IEnumerable<LS2V<float, Color>> plotLines,
            IEnumerable<RV<float, Color>> filledRects,
            IEnumerable<RV<float, Color>> openRects
        )
        {
            WbImageVm.ImageData = Id.MakeImageData(
                    plotPoints: plotPoints,
                    filledRects: filledRects,
                    openRects: openRects,
                    plotLines: plotLines
                );
        }

        public void SetRects(
                        R<float> boundingRect,
                        IEnumerable<RV<float, Color>> filledRects
                    )
        {
            SetData(
                        boundingRect: boundingRect,
                        plotPoints: Enumerable.Empty<P2V<float, Color>>(),
                        plotLines: Enumerable.Empty<LS2V<float, Color>>(),
                        filledRects: filledRects,
                        openRects: Enumerable.Empty<RV<float, Color>>()
                    );
        }

        public void SetPoints(
                R<float> boundingRect,
                IEnumerable<P2V<float, Color>> plotPoints
            )
        {
            SetData(
                        boundingRect: boundingRect,
                        plotPoints: plotPoints,
                        plotLines: Enumerable.Empty<LS2V<float, Color>>(),
                        filledRects: Enumerable.Empty<RV<float, Color>>(),
                        openRects: Enumerable.Empty<RV<float, Color>>()
                    );
        }


        public void SetData(
            R<float> boundingRect,
            IEnumerable<P2V<float, Color>> plotPoints,
            IEnumerable<LS2V<float, Color>> plotLines,
            IEnumerable<RV<float, Color>> filledRects,
            IEnumerable<RV<float, Color>> openRects
        )
        {
            WbImageVm.ImageData = Id.MakeImageDataAndClip(
                clipRegion: boundingRect,
                plotPoints: plotPoints,
                filledRects: filledRects,
                openRects: openRects,
                plotLines: plotLines
             );

        }

        private string _maxStrX;
        public string MaxStrX
        {
            get { return _maxStrX; }
            set { SetProperty(ref _maxStrX, value); }
        }

        private string _minStrX;
        public string MinStrX
        {
            get { return _minStrX; }
            set { SetProperty(ref _minStrX, value); }
        }

        private string _minStrY;
        public string MinStrY
        {
            get { return _minStrY; }
            set { SetProperty(ref _minStrY, value); }
        }

        private string _maxStrY;
        public string MaxStrY
        {
            get { return _maxStrY; }
            set { SetProperty(ref _maxStrY, value); }
        }

        private string _title;
        public string Title
        {
            get { return _title; }
            set { SetProperty(ref _title, value); }
        }

        private string _titleX;
        public string TitleX
        {
            get { return _titleX; }
            set { SetProperty(ref _titleX, value); }
        }

        private string _titleY;
        public string TitleY
        {
            get { return _titleY; }
            set { SetProperty(ref _titleY, value); }
        }

        private string _watermark;

        public string Watermark
        {
            get { return _watermark; }
            set { SetProperty(ref _watermark, value); }
        }
    }
}
