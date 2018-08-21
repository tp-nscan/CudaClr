using Sponge.Common;
using System.Windows;
using System;
using System.Windows.Media.Imaging;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Reactive.Subjects;

namespace Sponge.ViewModel.Common
{
    public class CameraVm : BindableBase
    {
        public IObservable<PngBitmapEncoder> OnScreenShot
        {
            get { return _onScreenShot; }
        }

        private readonly Subject<PngBitmapEncoder> _onScreenShot
            = new Subject<PngBitmapEncoder>();

        private Visibility _cameraVisibility;
        public Visibility CameraVisibility
        {
            get { return _cameraVisibility; }
            set
            {
                SetProperty(ref _cameraVisibility, value);
            }
        }

        private PngBitmapEncoder _pngBitmapEncoder;
        public PngBitmapEncoder PngBitmapEncoder
        {
            get { return _pngBitmapEncoder; }
            set
            {
                _pngBitmapEncoder = value;
                _onScreenShot.OnNext(value);
            }
        }

    }
}
