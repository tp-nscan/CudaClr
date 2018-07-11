using System.Windows.Media;
using Sponge.Common;
using FS;

namespace Sponge.ViewModel.Common
{
    public class TileVm : BindableBase
    {

        private R<float> _boundingRect;
        public R<float> BoundingRect
        {
            get { return _boundingRect; }
            set
            {
                SetProperty(ref _boundingRect, value);
                Width = value.MaxX - value.MinX;
                Height = value.MaxY - value.MinY;
            }
        }

        private double _width;
        public double Width
        {
            get { return _width; }
            set
            {
                SetProperty(ref _width, value);
            }
        }

        private double _height;
        public double Height
        {
            get { return _height; }
            set
            {
                SetProperty(ref _height, value);
            }
        }

        private Color _color;
        public Color Color
        {
            get { return _color; }
            set
            {
                SetProperty(ref _color, value);
            }
        }

        private string _textA;
        public string TextA
        {
            get { return _textA; }
            set
            {
                SetProperty(ref _textA, value);
            }
        }
        private string _textB;
        public string TextB
        {
            get { return _textB; }
            set
            {
                SetProperty(ref _textB, value);
            }
        }

        private string _textC;
        public string TextC
        {
            get { return _textC; }
            set
            {
                SetProperty(ref _textC, value);
            }
        }
    }
}
