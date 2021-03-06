﻿using System.Collections.Generic;
using System.Linq;
using System.Windows.Media;
using Sponge.Common;
using FS;

namespace Sponge.ViewModel.Common
{
    public class LegendVmClock : BindableBase
    {
        public static LegendVmClock Standard()
        {
            return new LegendVmClock("0.0", "0.5", "1.0", Colors.Pink, new Color[] {}, Colors.DeepPink );
        }

        public LegendVmClock(string minVal, string midVal, string maxVal, 
            Color minColor, Color[] midColors, Color maxColor)
        {
            WbImageVm = new WbImageVm();
            _minVal = minVal;
            _midVal = midVal;
            _maxVal = maxVal;
            _minColor = minColor;
            _midColors = midColors;
            _maxColor = maxColor;
            DrawMidColors();
        }

        public Color ColorVal(int value)
        {
            if (value < 0) return MinColor;
            if (value > ColorCount) return MaxColor;
            return MidColors[value];
        }

        void DrawMidColors()
        {
            WbImageVm.ImageData = Id.MakeImageData(
                plotPoints: Enumerable.Empty<P2V<float, Color>>(),
                filledRects: PlotRectangles,
                openRects: Enumerable.Empty<RV<float, Color>>(),
                plotLines: Enumerable.Empty<LS2V<float, Color>>());
        }

        IEnumerable<RV<float, Color>> PlotRectangles
        {
            get
            {
                return
                    MidColors.Select(
                        (c, i) => new RV<float, Color>(
                            minX: i + 1,
                            minY: 0,
                            maxX: i + 2,
                            maxY: 1,
                            v: c)
                        )
                        
                        //.Concat(new[]
                        //{
                        //    new RV<float, Color>(
                        //    minX: 0,
                        //    minY: 0,
                        //    maxX: 1,
                        //    maxY: 1,
                        //    v: MinCol)
                        //}).Concat(new[]
                        //{
                        //    new RV<float, Color>(
                        //    minX: MidColors.Length + 1,
                        //    minY: 0,
                        //    maxX: MidColors.Length + 2,
                        //    maxY: 1,
                        //    v: MaxColor)
                        //})
                        
                        ;
            }
        }

        public WbImageVm WbImageVm { get; }

        private string _minVal;
        public string MinVal
        {
            get { return _minVal; }
            set
            {
                SetProperty(ref _minVal, value);
            }
        }

        private string _midVal;
        public string MidVal
        {
            get { return _midVal; }
            set
            {
                SetProperty(ref _midVal, value);
            }
        }

        private string _maxVal;
        public string MaxVal
        {
            get { return _maxVal; }
            set
            {
                SetProperty(ref _maxVal, value);
            }
        }


        private Color _minColor;
        public Color MinColor
        {
            get { return _minColor; }
            set
            {
                SetProperty(ref _minColor, value);
            }
        }

        public int ColorCount { get; private set; }

        private Color[] _midColors;
        public Color[] MidColors
        {
            get { return _midColors; }
            set
            {
                SetProperty(ref _midColors, value);
                ColorCount = _midColors.Length;
                DrawMidColors();
            }
        }

        private Color _maxColor;
        public Color MaxColor
        {
            get { return _maxColor; }
            set
            {
                SetProperty(ref _maxColor, value);
            }
        }
    }
}
