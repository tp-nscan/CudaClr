using System;
using System.Collections.Generic;
using System.Linq;
using System.Reactive.Subjects;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Media;
using Sponge.Common;
using Utils;

namespace Sponge.ViewModel.Common
{
    public class ColorSequenceVm : BindableBase
    {
        private readonly Subject<Color[]> _colorsChanged
            = new Subject<Color[]>();
        public IObservable<Color[]> OnColorsChanged => _colorsChanged;


        public ColorSequenceVm(int colorCount, 
            double vred = 0.2, double vgreen = 0.2, double vblue = 0.2,
            double cred = 0.0, double cgreen = 0.333, double cblue = 0.666, 
            bool wrap = true)
        {
            ColorCount = colorCount;
            _wRed = vred;
            _wGreen = vgreen;
            _wBlue = vblue;
            _cRed = cred;
            _cGreen = cgreen;
            _cBlue = cblue;
            _wrap = wrap;
        }

        public int ColorCount { get; private set; }

        public void SendColorsChanged()
        {
            _colorsChanged.OnNext(
                ColorSeq.ColorTent(
                    length: ColorCount,
                    cR: CRed,
                    cG: CGreen,
                    cB: CBlue,
                    wR: WRed,
                    wG: WGreen,
                    wB: WBlue,
                    wrap: Wrap).ToArray()
                );
        }

        private bool _wrap;
        public bool Wrap
        {
            get { return _wrap; }
            set
            {
                SetProperty(ref _wrap, value);
                SendColorsChanged();
            }
        }

        private double _wBlue;
        public double WBlue
        {
            get { return _wBlue; }
            set
            {
                SetProperty(ref _wBlue, value);
                SendColorsChanged();
            }
        }


        private double _wGreen;
        public double WGreen
        {
            get { return _wGreen; }
            set
            {
                SetProperty(ref _wGreen, value);
                SendColorsChanged();
            }
        }

        private double _wRed;
        public double WRed
        {
            get { return _wRed; }
            set
            {
                SetProperty(ref _wRed, value);
                SendColorsChanged();
            }
        }


        private double _cBlue;
        public double CBlue
        {
            get { return _cBlue; }
            set
            {
                SetProperty(ref _cBlue, value);
                SendColorsChanged();
            }
        }


        private double _cGreen;
        public double CGreen
        {
            get { return _cGreen; }
            set
            {
                SetProperty(ref _cGreen, value);
                SendColorsChanged();
            }
        }

        private double _cRed;
        public double CRed
        {
            get { return _cRed; }
            set
            {
                SetProperty(ref _cRed, value);
                SendColorsChanged();
            }
        }

    }
}
