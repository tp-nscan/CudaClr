using System;
using System.Collections.Generic;
using System.Linq;
using System.Windows.Media;
using Sponge.Common;
using FS;

namespace Sponge.ViewModel.Common
{
    public class GraphLatticeVm : BindableBase
    {
        public GraphLatticeVm(R<uint> latticeBounds, string title= "Title", string titleX = "TitleX", string titleY = "TitleY")
        {
            _wbImageVm = new WbImageVm();
            _imageSize = new Sz2<double>(1.0, 1.0);
            _wbImageVm.ImageData = Id.InitImageData();
            LatticeBounds = latticeBounds;
            MinX = new IntRangeVm(min: (int)LatticeBounds.MinX, max: (int)LatticeBounds.MaxX, cur: (int)LatticeBounds.MinX);
            MinX.OnCurValChanged.Subscribe(v=>CurvalChanged());
            MaxX = new IntRangeVm(min: (int)LatticeBounds.MinX, max: (int)LatticeBounds.MaxX, cur: (int)LatticeBounds.MaxX);
            MaxX.OnCurValChanged.Subscribe(v => CurvalChanged());
            MinY = new IntRangeVm(min: (int)LatticeBounds.MinY, max: (int)LatticeBounds.MaxY, cur: (int)LatticeBounds.MinY);
            MinY.OnCurValChanged.Subscribe(v => CurvalChanged());
            MaxY = new IntRangeVm(min: (int)LatticeBounds.MinY, max: (int)LatticeBounds.MaxY, cur: (int)LatticeBounds.MaxY);
            MaxY.OnCurValChanged.Subscribe(v => CurvalChanged());

            Title = title;
            TitleX = titleX;
            TitleY = titleY;
        }

        void CurvalChanged()
        {
            MinX.MaxVal = MaxX.CurVal - 1;
            MaxX.MinVal = MinX.CurVal + 1;
            MinY.MaxVal = MaxY.CurVal - 1;
            MaxY.MinVal = MinY.CurVal + 1;
            Update();
        }

        private Func<P2<int>, R<double>, object> _cellUpdater;
        public Func<P2<int>, R<double>, object> GetCellUpdater()
        {
            return _cellUpdater;
        }

        public void SetUpdater(Func<P2<int>, R<double>, object> value)
        {
            _cellUpdater = value;
            if(WbImageVm != null) Update();
        }

        public void Update()
        {
            var results = new List<RV<float, Color>>();
            var cellSize = new Sz2<double>
             (
                x:ImageSize.X/(MaxX.CurVal - MinX.CurVal),
                y: ImageSize.X / (MaxY.CurVal - MinY.CurVal)
              );

            for(var i = MinX.CurVal; i < MaxX.CurVal; i++)
            {
                for (var j = MinY.CurVal; j < MaxY.CurVal; j++)
                {
                    results.Add( (RV<float, Color>)
                        _cellUpdater(new P2<int>(x: i, y: j),
                                    new R<double>(
                                            minX: i * cellSize.X,
                                            maxX: (i + 1) * cellSize.X,
                                            minY: j * cellSize.Y,
                                            maxY: (j + 1) * cellSize.Y
                                           )));
                }
            }

            WbImageVm.ImageData = Id.MakeImageData(
                    plotPoints: Enumerable.Empty<P2V<float, Color>>(),
                    plotLines: Enumerable.Empty<LS2V<float, Color>>(),
                    filledRects: results,
                    openRects: Enumerable.Empty<RV<float, Color>>()
                );
        }

        public R<uint> LatticeBounds { get; }

        private readonly WbImageVm _wbImageVm;
        public WbImageVm WbImageVm => _wbImageVm;

        private IntRangeVm _maxX;
        public IntRangeVm MaxX
        {
            get { return _maxX; }
            set
            {
                SetProperty(ref _maxX, value);
            }
        }

        private IntRangeVm _minX;
        public IntRangeVm MinX
        {
            get { return _minX; }
            set
            {
                SetProperty(ref _minX, value);
            }
        }

        private IntRangeVm _minY;
        public IntRangeVm MinY
        {
            get { return _minY; }
            set
            {
                SetProperty(ref _minY, value);
            }
        }

        private IntRangeVm _maxY;
        public IntRangeVm MaxY
        {
            get { return _maxY; }
            set
            {
                SetProperty(ref _maxY, value);
            }
        }

        private string _title;
        public string Title
        {
            get { return _title; }
            set
            {
                SetProperty(ref _title, value);
            }
        }

        private string _titleX;
        public string TitleX
        {
            get { return _titleX; }
            set
            {
                SetProperty(ref _titleX, value);
            }
        }

        private string _titleY;
        public string TitleY
        {
            get { return _titleY; }
            set
            {
                SetProperty(ref _titleY, value);
            }
        }


        private Sz2<double> _imageSize;
        public Sz2<double> ImageSize
        {
            get { return _imageSize; }
            set
            {
                SetProperty(ref _imageSize, value);
            }
        }
    }
}
